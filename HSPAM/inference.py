
import argparse
import glob
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.color import rgb2lab
from skimage.segmentation import mark_boundaries
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import normalize_lab, denormalize_lab
from datasets import BSD

import torch.nn.functional as F
import ssn.enforce_connectivity_mask
from ssn._enforce_connectivity_mask import _enforce_connectivity_labels_with_mask
from skimage.segmentation._slic import _enforce_label_connectivity_cython
import matplotlib.colors as mcolors

from ssn.model import create_model
from ssn.utils import (
    precompute_abs_indices,
    compute_inputs,
    get_majority_object,
    verif_segmentation_in_mask,
    show_objects_with_background,
    dilate_object_map,
    get_majority_object_with_hist,
    dilate_object_map_K1,
    check_background_dino,
)
from collections import defaultdict

from SAM.sam_segment_preprocessing import (
    segment_with_sam_main,
    segment_with_fastsam_main,
)
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from hierarchy.get_hierarchy import get_global_hierarchy, get_intermediate_clusters
from utils.utils_inference import  achievable_segmentation_accuracy, default_on_result_saver,  keep_largest_objects_by_label, interactive_attention_map

VERBOSE = False


@torch.no_grad()
def inference_one_image(
    inputs,
    nspix,
    color_scale=0.26,
    pos_scale=7.5,
    mask=None,
    enforce_connectivity=True,
    model=None,
    args=None,
    img_clean=None
):

    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    height, width = inputs.shape[-2:]

    mask = keep_largest_objects_by_label(mask, nspix)
    tic = time.time()
    # Precompute maps & stuff
    (
        pre_label_map,
        abs_indices,
        object_map,
        lab_vect,
        nspix_ratio_map,
        centers,
        n_sp_vect,
    ) = precompute_abs_indices(inputs, nspix, 1, device, mask, ratio=args.ratio, use_dino=args.use_dino)

    inputs, coords, inputs_weighted, coords_weighted = compute_inputs(
        inputs,
        nspix,
        pos_scale,
        color_scale,
        1,
        nspix_ratio_map,
        normalize=False,
        device=device,
    )

    # MaskSSN forward
    _, H, _, pixel_features = model(
        inputs, pre_label_map, abs_indices, inputs_weighted, coords_weighted
    )
    labels_pre_enforce = H.reshape(height, width).to("cpu").detach().numpy()
    if VERBOSE:
        toc = time.time()
        print(f"Time to segment: {toc - tic:.2f} s")
        tic = time.time()
    object_map = np.argmax(
        np.concatenate((np.zeros((1, height, width)), object_map)), axis=0
    )
    artefacts = (object_map == 0).astype("int")
    final_object_map = get_majority_object(
        labels_pre_enforce.astype("int"), lab_vect)

    if enforce_connectivity:
        segment_map = height * width / (nspix * nspix_ratio_map)
        min_size_map = (0.15 * segment_map).astype("int")  # min from SAM
        max_size_map = (3.0 * segment_map).astype("int")
        max_size = np.max(max_size_map)

        if VERBOSE:
            print("before enforce {}", len(np.unique(labels_pre_enforce)))

        labels_mask = _enforce_connectivity_labels_with_mask(
            labels_pre_enforce[None],
            final_object_map[np.newaxis, :, :],
            artefacts[np.newaxis, :, :],
            min_size_map[np.newaxis, :, :],
            max_size_map[np.newaxis, :, :],
            max_size,
            n_sp_vect[:],
        )[0]

        if VERBOSE:
            print("after enforce {}", len(np.unique(labels_mask)))

    toc = time.time()
    if VERBOSE:
        print(f"Time to enforce connectivity: {toc - tic:.2f} s")

    final_object_map = get_majority_object_with_hist(
        labels_mask, final_object_map)

    verif_segmentation_in_mask(
        labels_mask.astype("int"), final_object_map + 1)
    return labels_mask, final_object_map,  pixel_features, inputs

def get_nspix_hierarchy(labels_mask, pixel_features=None, inputs=None, object_map=None, nspix=None, args=None, attention_map=None):
    logs = {}
    Z = get_global_hierarchy(labels_mask,
                             pixel_features=pixel_features,
                             w_pos=args.w_pos,
                             inputs=inputs,
                             object_map=object_map,
                             object_merge_mode=args.object_merge_mode,
                             attention_map=attention_map,
                             w_att=args.w_att,
                             attention_pair_mode=args.attention_pair_mode,
                             attention_scope=args.attention_scope,
                             threshold_attention=args.threshold_attention)
    hierarchy_results = []
    if "index_change_phase" in Z:
        logs["index_change_phase"] = Z["index_change_phase"]
    for nb_sp in nspix:
        constrained_superpixels = get_intermediate_clusters(
            Z, labels_mask, n_clusters=nb_sp
        )
        hierarchy_results.append(constrained_superpixels)
    return hierarchy_results, logs

def inference_image(args):
    image = plt.imread(args.image)
    if image.dtype == "float32":
        image = (image * 255).astype("uint8")

    s = time.time()

    object_mask = None
    if args.mask:
        object_mask = np.load(args.mask)

    # Optional SAM/FastSAM preprocessing
    if args.use_sam:
        object_mask = segment_with_sam_main(
            image,
            "grid",
            points_per_side=args.points_per_side,
        )
    if args.use_fastsam:
        object_mask = segment_with_fastsam_main(
            image, postprocess=True
        )

    if (args.use_sam or args.mask) and args.dilate:
        object_mask = dilate_object_map(object_mask, args.dilate)

    model = create_model(
        args.type_model,
        args.weight,
        args.checkpoint,
        args.fdim,
        args.nspix,
        args.niter,
        use_pca=args.use_pca,
    )

    image_torch = (
        torch.tensor(rgb2lab(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .type(torch.FloatTensor)
        .cuda()
    )

    superpixel_map, final_object_map, pixel_features, inputs = inference_one_image(
        image_torch,
        args.nspix,
        args.color_scale,
        args.pos_scale,
        object_mask,
        model=model,
        args=args,
        img_clean=image,
    )
    res_dest = os.path.join(output_path, "output_one_image")
    os.makedirs(res_dest, exist_ok=True)
    all_attention = None
    path_attention_map = os.path.join(res_dest, "attention_map")
    os.makedirs(path_attention_map, exist_ok=True)
    if args.w_att > 0:
        if args.interactive:
            print(
                "Interactive mode activated : using user attention map (no DINO).")
            args.use_dino = False
            clicks_overlay_path = os.path.join(
                path_attention_map, f"{os.path.basename(os.path.normpath(args.image[0:-4]))}_attention_clicks_overlay.png"
            )
            all_attention = interactive_attention_map(
                image, final_object_map, save_path=clicks_overlay_path
            )
            if all_attention.shape != superpixel_map.shape:
                all_attention = cv2.resize(
                    all_attention.astype(np.float32),
                    (superpixel_map.shape[1], superpixel_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            inputs_torch = torch.from_numpy(image).unsqueeze(
                0).float().permute(0, 3, 1, 2).cuda()
            _, all_attention = check_background_dino(
                inputs_torch, final_object_map)
            if all_attention is not None and all_attention.shape != superpixel_map.shape:
                all_attention = F.interpolate(
                    all_attention.unsqueeze(0).unsqueeze(0), size=superpixel_map.shape, mode="nearest")[0, 0]

    object_map_hierarchy = final_object_map 
        
    Z = get_global_hierarchy(
        superpixel_map,
        pixel_features=pixel_features,
        w_pos=args.w_pos,
        inputs=inputs,
        object_map=object_map_hierarchy,
        object_merge_mode=args.object_merge_mode,
        attention_map=all_attention,
        w_att=args.w_att,
        attention_pair_mode=args.attention_pair_mode,
        attention_scope=args.attention_scope,
        threshold_attention=args.threshold_attention)

    attention_map_img= Z["attn_region_map"]
    for i_scale in range(1,args.nspix):
        print(f"Processing scale {i_scale}")
        label_i = get_intermediate_clusters(Z, superpixel_map, n_clusters=i_scale)
        path_label = os.path.join(res_dest, "label",os.path.basename(os.path.normpath(args.image[0:-4]))+ "_" + str(i_scale) + "_label.npy")
        os.makedirs(os.path.dirname(path_label), exist_ok=True)
        np.save(path_label, label_i)
        path_img = os.path.join(res_dest, "img", os.path.basename(os.path.normpath(args.image[0:-4]))+ "_" + str(i_scale) + ".png")
        os.makedirs(os.path.dirname(path_img), exist_ok=True)
        plt.imsave(path_img, mark_boundaries(image, label_i))

    if args.w_att > 0:
        path_attention_map_img = os.path.join(path_attention_map, f"{os.path.basename(os.path.normpath(args.image[0:-4]))}_attention_map.png")
        plt.imsave(path_attention_map_img, attention_map_img)

def get_object_mask(img_clean_np, args, img_name, sam_model=None):
    object_mask = None
    if getattr(args, "folder_mask", None):
        mask_path = os.path.join(args.folder_mask, img_name + ".npy")
        if os.path.exists(mask_path):
            object_mask = np.load(mask_path)

    if object_mask is None:
        if args.use_sam:
            object_mask = segment_with_sam_main(
                img_clean_np,
                "grid",
                points_per_side=args.points_per_side,
                sam_model=sam_model,
            )
        elif args.use_fastsam:
            object_mask = segment_with_fastsam_main(
                img_clean_np, postprocess=True
            )

    if (args.use_sam or args.use_fastsam or args.mask) and args.dilate:
        object_mask = (
            dilate_object_map_K1(object_mask)
            if args.dilate == 1
            else dilate_object_map(object_mask, args.dilate)
        )
    return object_mask


def inference_folder(args, np_range, on_result=None):
    ext = (".jpg", ".png", ".jpeg")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path_imgs_test = os.path.join(args.folder, "images", "test")
    image_paths = sorted(
        [
            os.path.join(path_imgs_test, f)
            for f in os.listdir(path_imgs_test)
            if f.lower().endswith(ext)
        ]
    )

    test_dataset = BSD(root=args.folder, split="test")
    test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)


    model = create_model(
        args.type_model,
        args.weight,
        args.checkpoint,
        args.fdim,
        args.nspix,
        args.niter,
        use_pca=args.use_pca,
    )
    for i, data in enumerate(tqdm(test_loader)):
        img_name = os.path.basename(os.path.normpath(image_paths[i][0:-4]))
        inputs, labels_gt, edges, edges_mask, img_clean = data

        inputs = inputs.to(device)
        labels_gt = labels_gt.to(device)

        img_clean_np = (
            img_clean[0].cpu().detach().numpy() * 255).astype(np.uint8)

        object_mask = get_object_mask(img_clean_np, args, img_name)
        object_mask = keep_largest_objects_by_label(object_mask, args.nspix)
        superpixel_map, final_object_map, pixel_features, inputs = inference_one_image(
            inputs,
            args.nspix,
            args.color_scale,
            args.pos_scale,
            object_mask,
            model=model,
            args=args,
            img_clean=img_clean_np,
        )
        all_attention = None
        if args.w_att > 0:
            inputs_torch = torch.from_numpy(img_clean_np).unsqueeze(
                0).float().permute(0, 3, 1, 2).to(device)
            _, all_attention = check_background_dino(
                inputs_torch, final_object_map)
            if all_attention is not None and all_attention.shape != superpixel_map.shape:
                all_attention = F.interpolate(
                    all_attention.unsqueeze(0).unsqueeze(0),
                    size=superpixel_map.shape,
                    mode="nearest",
                )[0, 0]
        hierarchy_results, logs = get_nspix_hierarchy(superpixel_map,
            pixel_features=pixel_features,
            inputs=inputs,
            object_map=final_object_map,
            nspix=args.np_range,
            args=args,
            attention_map=all_attention
        )
        for i_scale, label_i in enumerate(hierarchy_results):
            output_path = os.path.join(args.out_folder, str(args.np_range[i_scale]))
            dir_path_label = os.path.join(output_path, "label")
            dir_path_img = os.path.join(output_path, "img")
            dir_path_sam = os.path.join(output_path, "sam_map")
            for directory in (dir_path_label, dir_path_img, dir_path_sam):
                os.makedirs(directory, exist_ok=True)

            res_dest_label = os.path.join(dir_path_label,f"{img_name}_{np_range[i_scale]}_label.npy")
            res_dest_sam_map_img = os.path.join(dir_path_sam, f"{img_name}_sam_final_map_2.png")
            res_dest_img = os.path.join(dir_path_img, f"{img_name}_{args.pos_scale}_{args.color_scale}.png")

            asa_sp_map = 0.0
            for _, label_gt in enumerate(labels_gt[0]):
                gt_np = label_gt.cpu().detach().numpy()
                asa_sp_map += achievable_segmentation_accuracy(label_i, gt_np)
            asa_sp_map /= len(labels_gt[0])

            print(f"ASA: {asa_sp_map}")
            nb_sp = len(np.unique(label_i))
            print(f"Number of superpixels: {nb_sp} vs {np_range[i_scale]}")

            np.save(res_dest_label, label_i)
            plt.imsave(res_dest_img, mark_boundaries(
                img_clean_np, label_i))
            plt.imsave(res_dest_sam_map_img, final_object_map)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=None,
        type=str,
        help="/path/to/image",
    )
    parser.add_argument(
        "--folder", default=None, type=str, help="/path/to/folder containing images"
    )
    parser.add_argument(
        "--weight",
        default=None,
        type=str,
        help="/path/to/pretrained_weight",
    )
    parser.add_argument("--fdim", default=20, type=int,
                        help="embedding dimension")
    parser.add_argument(
        "--niter",
        default=10,
        type=int,
        help="number of iterations for differentiable SLIC",
    )
    parser.add_argument("--nspix", default=1250, type=int,
                        help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=7.5, type=float)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default="./models/spam_checkpoint.pt",
        type=str,
        help="pretrained model (weights + optimizer state) to load",
    )
    parser.add_argument(
        "--mask",
        default=None,
        type=str,
        help="integer mask to constrain the segmentation",
    )
    parser.add_argument(
        "--folder_mask",
        default=None,
        type=str,
        help="integer masks folder to constrain the segmentation",
    )
    parser.add_argument(
        "--out_folder",
        default="./res",
        type=str,
        help="output path to save results",
    )
    parser.add_argument(
        "--use_sam", action="store_true", help="recompute SAM on the fly"
    )
    parser.add_argument(
        "--use_fastsam", action="store_true", help="recompute SAM on the fly"
    )
    parser.add_argument("--points_per_side", default=32,
                        type=int, help="8, 16, 32, 64")
    parser.add_argument(
        "--dilate", default=0, type=int, help="object border dilatation (3, 5, ...)"
    )
    parser.add_argument(
        "--type_model",
        default="ssn",
        type=str,
        help="ssn, resnet50, resnet101, deeplabv3",
    )
    parser.add_argument(
        "--use_pca", action="store_true", help="Use pca for feature reduction"
    )
    parser.add_argument(
        "--override_results", action="store_true", help="Override results"
    )
    parser.add_argument(
        "--use_dino", action="store_true", help="Use dino for background detection"
    )
    parser.add_argument(
        "--ratio", default=1, type=float, help="ratio for superpixel size"
    )
    parser.add_argument(
        "--object_merge_mode", default="relaxed", type=str, help="object merge mode"
    )
    parser.add_argument(
        "--w_att", default=0, type=float, help="attention weight"
    )
    parser.add_argument(
        "--attention_pair_mode", default="max", type=str, help="attention pair mode"
    )
    parser.add_argument(
        "--attention_scope", default="object", type=str, help="attention scope"
    )
    parser.add_argument(
        "--enforce_connectivity", default=True, type=bool, help="enforce connectivity"
    )
    parser.add_argument(
        "--threshold_attention", default=None, type=float, help="threshold attention"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive iterative refinement"
    )
    parser.add_argument(
        "--np_range", nargs="+", default=[50, 150, 250, 350,500,650,800,1000,1250], type=int, help="number of superpixels"
    )
    parser.add_argument(
        "--w_pos", default=5, type=float, help="w_pos"
    )
    args = parser.parse_args()

    output_path = args.out_folder


    np_range = args.np_range
    if args.use_sam and args.use_fastsam:
        raise ValueError("--use_sam and --use_fastsam cannot be used together")

    if args.interactive and args.folder != None:
        raise ValueError("--interactive and --folder cannot be used together")

    if args.image != None:
        inference_image(args)
    elif args.folder != None:
        saver = default_on_result_saver(
            args.out_folder, args.pos_scale, args.color_scale, args.override_results
        )
        asa_dataframe = inference_folder(args, args.np_range, on_result=saver)
        asa_dataframe.to_csv(os.path.join(
            args.out_folder, "asa.csv"), index=False)
    else:
        print("No image or folder given")
