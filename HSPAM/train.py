import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from skimage.color import lab2rgb
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

from datasets import BSD, augmentations
from ssn.model import create_model
from ssn.loss import (
    reconstruct_loss_with_cross_entropy,
    reconstruct_loss_with_mse,
    contour_loss,
    F_contour_loss,
)
from ssn.utils import (
    Meter,
    map_from_masks_ratio,
    get_mask_centroids,
    get_abs_indices,
    precompute_abs_indices,
    compute_inputs,
    labelmap2onehot,
)


@torch.no_grad()
def eval(model, loader, color_scale, pos_scale, normalize, device):
    def achievable_segmentation_accuracy(superpixel, label):
        """
        Function to calculate Achievable Segmentation Accuracy:
            ASA(S,G) = sum_j max_i |s_j \cap g_i| / sum_i |g_i|

        Args:
            input: superpixel image (H, W),
            output: ground-truth (H, W)
        """
        TP = 0
        unique_id = np.unique(superpixel)
        for uid in unique_id:
            mask = superpixel == uid
            label_hist = np.histogram(label[mask])
            maximum_regionsize = label_hist[0].max()
            TP += maximum_regionsize
        return TP / label.size

    model.eval()
    sum_asa = 0
    sum_nsp = 0
    ii = 0
    for data in loader:

        inputs, labels, _, _, _ = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Inputs
        batchsize, feat, height, width = inputs.shape
        # Precompute maps & stuff
        (
            pre_label_map,
            abs_indices,
            object_map,
            lab_vect,
            nspix_ratio_map,
            centers,
            n_sp_vect,
        ) = precompute_abs_indices(inputs, model.n_sp, batchsize, device, None)
        inputs_model, _, inputs_weighted, coords_weighted = compute_inputs(
            inputs,
            model.n_sp,
            pos_scale,
            color_scale,
            batchsize,
            nspix_ratio_map,
            normalize,
            device,
        )

        # MaskSSN
        _, H, _, _ = model(
            inputs_model, pre_label_map, abs_indices, inputs_weighted, coords_weighted
        )

        H = H.reshape(batchsize, height, width)
        labels = labels.argmax(1).reshape(batchsize, height, width)

        asa = achievable_segmentation_accuracy(
            H.to("cpu").detach().numpy(), labels.to("cpu").numpy()
        )
        sum_asa += asa

        ii = ii + 1
        for i in range(0, H.shape[0]):
            image_i = np.transpose(inputs[i].to(
                "cpu").detach().numpy(), (1, 2, 0))
            labels = H[i].to("cpu").detach().numpy()
            segment_size = height * width / model.n_sp
            min_size = int(0.15 * segment_size)
            max_size = int(3.0 * segment_size)
            labels = _enforce_label_connectivity_cython(
                labels[None], min_size, max_size, 0
            )[0]
            plt.imsave(
                "logs/res_eval_" + str(ii) + ".png",
                mark_boundaries(
                    (lab2rgb((image_i)) * 255).astype("uint8"), labels),
            )
            sum_nsp += np.max(labels)
    model.train()
    return sum_asa / len(loader), sum_nsp / len(loader)


def update_param(
    data,
    model,
    optimizer,
    compactness,
    color_scale,
    pos_scale,
    normalize,
    device,
    init_label_map,
    abs_indices,
    nspix_ratio_map,
    loss_type="seg",
    use_sam=False,
):

    inputs, labels, edges, edges_mask, _ = data

    image = np.copy(inputs)
    inputs = inputs.to(device)
    labels = labels.to(device)
    edges = edges.to(device)
    edges_mask = edges_mask.to(device)

    batchsize, _, height, width = inputs.shape

    if use_sam:  # Recalcul de init_label_map et abs_spix_indices
        abs_spix_indices = abs_indices[1]
        for i in range(0, batchsize):
            label_map = objects_maps[i].detach().cpu().numpy()
            onehot = labelmap2onehot(label_map)
            onehot, artefacts = onehot[1:], onehot[0]
            init_label_map_i, centers, lab_vect, n_sp_vect = map_from_masks(
                label_map, sp_size=model.n_sp
            )
            abs_spix_indices_i = get_abs_indices(
                init_label_map_i, artefacts, centers, lab_vect
            )
            abs_spix_indices_i = abs_spix_indices_i.repeat(1, 1, 1)
            init_label_map_i = init_label_map_i.repeat(1, 1, 1, 1)

            abs_spix_indices[i] = abs_spix_indices_i
            init_label_map[i] = init_label_map_i

        abs_indices[1] = abs_spix_indices

    inputs, coords_normalized, inputs_weighted, coords_weighted = compute_inputs(
        inputs,
        model.n_sp,
        pos_scale,
        color_scale,
        batchsize,
        nspix_ratio_map,
        normalize,
        device,
    )

    Q, H, _, _ = model(
        inputs, init_label_map, abs_indices, inputs_weighted, coords_weighted
    )

    # Loss
    recons_loss = 0
    cont_loss = 0
    loss = 0

    compact_loss = reconstruct_loss_with_mse(
        Q, coords_weighted.reshape(*coords_weighted.shape[:2], -1), H
    )

    image_i = lab2rgb(np.transpose(image[0], (1, 2, 0)))

    # Contours
    if loss_type == "contours":
        cont_loss = F_contour_loss(Q, edges, edges_mask, mode="max")
        loss = compactness * compact_loss + cont_loss / 2

    # SEG
    if loss_type == "seg":
        recons_loss = reconstruct_loss_with_cross_entropy(Q, labels)
        loss = compactness * compact_loss + recons_loss

    # SEG + Contours
    if loss_type == "seg_contours":
        recons_loss = reconstruct_loss_with_cross_entropy(Q, labels)
        cont_loss = F_contour_loss(Q, edges, edges_mask, mode="max")
        loss = compactness * compact_loss + cont_loss / 5 + recons_loss / 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if loss_type == "seg":
        return {
            "loss": loss.item(),
            "compact": compact_loss.item(),
            "reconstruction": recons_loss.item(),
        }
    if loss_type == "contours":
        return {
            "loss": loss.item(),
            "compact": compact_loss.item(),
            "contour": cont_loss.item(),
        }
    if loss_type == "seg_contours":
        return {
            "loss": loss.item(),
            "compact": compact_loss.item(),
            "reconstruction": recons_loss.item(),
            "contour": cont_loss.item(),
        }
    return {}


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = create_model(
        cfg.type_model,
        cfg.weights,
        cfg.model,
        cfg.fdim,
        cfg.nspix,
        cfg.niter,
        use_pca=False,  # use_pca is not used for training
    )
    model.train()
    optimizer = optim.Adam(model.parameters(), cfg.lr)

    augment = augmentations.Compose(
        [
            augmentations.RandomHorizontalFlip(),
            augmentations.RandomScale(),
            augmentations.RandomCrop(),
        ]
    )
    train_dataset = BSD(cfg.root, geo_transforms=augment,
                        normalize_img=cfg.normalize)
    train_loader = DataLoader(
        train_dataset,
        cfg.batchsize,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.nworkers,
    )

    test_dataset = BSD(cfg.root, split="val", normalize_img=cfg.normalize)
    test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)

    meter = Meter()

    for data in train_loader:
        img, _, _, _, _ = data  # get first image for the shape
        break

    # Precompute maps & stuff
    (
        pre_label_map,
        abs_indices,
        object_map,
        lab_vect,
        nspix_ratio_map,
        centers,
        n_sp_vect,
    ) = precompute_abs_indices(img, cfg.nspix, cfg.batchsize, device, None)

    iterations = 0
    max_val_asa = 0
    while iterations < cfg.train_iter:
        for data in train_loader:
            iterations += 1
            metric = update_param(
                data,
                model,
                optimizer,
                cfg.compactness,
                cfg.color_scale,
                cfg.pos_scale,
                cfg.normalize,
                device,
                pre_label_map,
                abs_indices,
                nspix_ratio_map,
                cfg.loss_type,
                cfg.use_sam,
            )
            meter.add(metric)
            state = meter.state(f"[{iterations}/{cfg.train_iter}]")
            print(state)
            if (iterations % cfg.test_interval) == 0:
                asa, nsp = eval(
                    model,
                    test_loader,
                    cfg.color_scale,
                    cfg.pos_scale,
                    cfg.normalize,
                    device,
                )

                print(f"validation asa {asa}")
                if asa > max_val_asa:
                    max_val_asa = asa
                    state = {
                        "iter": iterations,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    torch.save(
                        state,
                        os.path.join(
                            cfg.out_dir,
                            "best_model_"
                            + cfg.loss_type
                            + str(iterations)
                            + "_ASA="
                            + str(asa)
                            + "_nsp="
                            + str(nsp)
                            + ".pt",
                        ),
                    )

                if (iterations % (cfg.test_interval * 10)) == 0:
                    state = {
                        "iter": iterations,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    torch.save(
                        state,
                        os.path.join(
                            cfg.out_dir,
                            "model_" + cfg.loss_type + str(iterations) + ".pt",
                        ),
                    )

            if iterations == cfg.train_iter:
                break

    unique_id = str(int(time.time()))
    torch.save(
        model.state_dict(), os.path.join(cfg.out_dir, "model" + unique_id + ".pth")
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, help="/path/to/BSR")
    parser.add_argument(
        "--out_dir", default="./logs", type=str, help="/path/to/output directory"
    )
    parser.add_argument("--batchsize", default=6, type=int)
    parser.add_argument(
        "--nworkers", default=4, type=int, help="number of threads for CPU parallel"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--train_iter", default=500000, type=int)
    parser.add_argument("--fdim", default=20, type=int,
                        help="embedding dimension")
    parser.add_argument(
        "--niter",
        default=5,
        type=int,
        help="number of iterations for differentiable SLIC",
    )
    parser.add_argument("--nspix", default=100, type=int,
                        help="number of superpixels")

    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=7.5, type=float)
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--compactness", default=1e-5, type=float)
    parser.add_argument("--test_interval", default=5000, type=int)
    parser.add_argument(
        "--weights", default=None, type=str, help="pretrained weights to load"
    )
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        help="pretrained model (weights + optimizer state) to load",
    )
    parser.add_argument(
        "--loss_type",
        default="seg",
        type=str,
        help="pretrained model (weights + optimizer state) to load",
    )
    parser.add_argument(
        "--use_sam", action="store_true", help="use SAM masks during training"
    )
    parser.add_argument(
        "--type_model",
        default="ssn",
        type=str,
        help="ssn, resnet50, resnet101, deeplabv3",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train(args)
