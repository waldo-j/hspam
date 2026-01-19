import torch
import numpy as np
import warnings
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
import math
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import DINO.inference_dino as dino
import torchvision.transforms as transforms
from skimage.color import lab2rgb
import os
import torch.nn.functional as F

VERBOSE = False


def dilate_object_map(L, kernel_size=3):

    background = L == 0
    k = kernel_size // 2

    boundary_mask = np.zeros_like(L, dtype=bool)
    L_dil = np.copy(L)
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(L, shift=dx, axis=0), shift=dy, axis=1)
            different_labels = (L != shifted) & (L != 0) & (shifted != 0)
            boundary_mask |= different_labels

    boundary_mask[background] = 1

    L_dil[boundary_mask] = 0
    return L_dil


def dilate_object_map_K1(L, kernel_size=3):
    k = kernel_size // 2
    h, w = L.shape
    L_dil = np.copy(L)
    L_shift = np.zeros_like(L)

    L_shift[:-1, :] = L[:-1, :] - L[1:, :]
    L_dil[np.logical_and(L_shift, L_shift != L)] = 0
    L_shift[:, :-1] = L[:, :-1] - L[:, 1:]
    L_dil[np.logical_and(L_shift, L_shift != L)] = 0

    L_dil[L == 0] = 0

    return L_dil


def show_objects_with_background(
    object_map,
    image_rgb,
    alpha=0.25,
    color_artefacts=[1, 1, 0],
    sp_color=[0, 0, 0],
    superpixels=None,
):

    # To have the same color labels
    np.random.seed(2)
    if np.min(object_map):  # if object map (final) does not have artefacts
        np.random.random(3)

    plt.figure(666)

    img = np.copy(image_rgb[0]).astype(np.float32) / 255
    img = img.transpose(1, 2, 0)
    colored_mask = np.zeros_like(img, dtype=np.float32)
    for uid in np.unique(object_map):
        m = object_map == uid
        color_mask = np.random.random(3)
        for i in range(3):
            colored_mask[..., i] += color_mask[i] * m
        img[m == 1] = colored_mask[m == 1] * alpha + (1 - alpha) * img[m == 1]

    if superpixels is not None:
        img = mark_boundaries(img, superpixels, sp_color)

    img[object_map == 0] = color_artefacts

    plt.imshow(img)
    plt.axis("off")

    return img


def verif_segmentation_in_mask(superpixel, object_map, test_ok=True):

    final_object_map = np.zeros((object_map.shape[0], object_map.shape[1])).astype(
        "int"
    )

    error = 0
    min_ = 0
    max_ = np.max(object_map)
    len_ = max_ + 1
    unique_id = np.unique(superpixel)
    for uid in unique_id:
        mask = superpixel == uid
        label_hist = np.histogram(
            object_map[mask], range=(min_, max_), bins=len_)
        label_hist = label_hist[0][1:]  # cut artefacts
        # If superposition with several objects !
        if np.sum(label_hist) != np.max(label_hist):
            if test_ok:
                print("ERROR OVERLAP OBJECTS")
                print(uid)
                print(np.sum(label_hist))
                print(np.max(label_hist))

            error = 1
        max_index = np.argmax(label_hist)
        final_object_map[mask] = max_index + 1

    if error == 0 and test_ok:
        if VERBOSE:
            print("OK ! Mask borders are respected")

    return final_object_map


# old version of get_majority_object with histogram
def get_majority_object_with_hist(superpixel, object_map):

    final_object_map = np.zeros((object_map.shape[0], object_map.shape[1])).astype(
        "int"
    )

    error = 0
    min_ = 0
    max_ = np.max(object_map)
    len_ = max_ + 1
    unique_id = np.unique(superpixel)
    for uid in unique_id:
        mask = superpixel == uid
        label_hist = np.histogram(
            object_map[mask], range=(min_, max_), bins=len_)
        label_hist = label_hist[0]  # cut artefacts
        # If superposition avec plusieurs objets !
        max_index = np.argmax(label_hist)
        final_object_map[mask] = max_index

    return final_object_map


def get_majority_object(superpixel, lab_vect, verif=False):

    final_object_map = np.zeros((superpixel.shape[0], superpixel.shape[1])).astype(
        "int"
    )
    for i in range(np.max(superpixel) + 1):
        final_object_map[superpixel == i] = lab_vect[i]

    return final_object_map


def compute_inputs(
    inputs,
    nspix,
    pos_scale,
    color_scale,
    batchsize,
    nspix_ratio_map,
    normalize=False,
    device="cuda",
):

    height, width = inputs.shape[-2:]

    nspix_per_axis = int(math.sqrt(nspix))
    nspix_ratio_map = torch.from_numpy(np.sqrt(nspix_ratio_map)).to(device)
    pos_scale_ = pos_scale * \
        max(nspix_per_axis / height, nspix_per_axis / width)

    # Generate XY coordinates and apply pos_scale weight
    Y, X = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device)
    )  # , indexing='ij') #FIXME: Indexing pour toutes les versions
    if normalize:
        print("cvsdcd")
        Y = Y / height
        X = X / width
        Y_weighted = Y * nspix_per_axis * np.sqrt(nspix_ratio_map)
        X_weighted = X * nspix_per_axis * np.sqrt(nspix_ratio_map)
        coords_weighted = torch.stack((Y_weighted, X_weighted), axis=0)
        coords_weighted *= pos_scale
        coords_weighted = coords_weighted[None].repeat(
            batchsize, 1, 1, 1).float()
    else:
        # Y = Y * pos_scale * nspix_per_axis / height * nspix_ratio_map
        # X = X * pos_scale * nspix_per_axis / width * nspix_ratio_map
        Y = Y * pos_scale_ * (nspix_ratio_map)
        X = X * pos_scale_ * (nspix_ratio_map)
    if VERBOSE:
        print("max XY")
        print(np.max(X.detach().cpu().numpy()))
        print(np.max(Y.detach().cpu().numpy()))
        print(f"{pos_scale}")
        print(f"{pos_scale_}")
        print(f"{nspix_per_axis}")
        print(np.min(nspix_ratio_map.detach().cpu().numpy()))
        print(np.max(nspix_ratio_map.detach().cpu().numpy()))
    coords = torch.stack((Y, X), 0)
    coords = coords[None].repeat(batchsize, 1, 1, 1).float()

    # Apply color_scale weight
    if VERBOSE:
        print("max inputs")
        print(np.min(inputs.detach().cpu().numpy()))
        print(np.max(inputs.detach().cpu().numpy()))
        print(f"{color_scale}")
    inputs_weighted = inputs * color_scale

    if 0:
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(X.detach().cpu().numpy())
        plt.subplot(122)
        plt.imshow(Y.detach().cpu().numpy())
        plt.show()
        print(inputs.mean(axis=[0, 2, 3]))
        print(inputs_weighted.mean(axis=[0, 2, 3]))
        print(coords.mean(axis=[0, 2, 3]))
        if normalize:
            print(coords_weighted.mean(axis=[0, 2, 3]))

    if normalize:
        inputs_coords = torch.cat([inputs, coords], axis=1)
        return inputs_coords, coords, inputs_weighted, coords_weighted
    else:
        inputs_coords = torch.cat([inputs_weighted, coords], axis=1)
        return inputs_coords, coords, inputs_weighted, coords


def precompute_abs_indices(
    inputs, nspix, batchsize, device="cuda", mask=None, ratio=1, use_dino=False, map_ratio=None
):

    height, width = inputs.shape[-2:]
    n_pixel = height * width
    object_map = np.ones((1, height, width))
    artefacts = np.zeros((1, height, width))
    # is_background = np.zeros((2), dtype=int)  # 0=artefacts, 1=objects
    ind_to_one_hot = np.zeros((1), dtype=int)
    if mask is not None:
        onehot, ind_to_one_hot = labelmap2onehot(mask)
        if np.min(mask) == 0:  # artefacts
            object_map, artefacts, ind_to_one_hot = (
                onehot[1:],
                onehot[0],
                ind_to_one_hot[1:],
            )

        else:
            object_map = onehot

        if use_dino:
            is_background,_ = check_background_dino(inputs, mask)
        else:
            N = np.max(mask)
            is_background = np.zeros(N + 1, dtype=int)
    else:
        is_background = np.zeros((2), dtype=int)  # 0=artefacts, 1=objects

    pre_label_map, centers, lab_vect, nspix_ratio_map, n_sp_vect = map_from_masks_ratio2(
        object_map, is_background, ind_to_one_hot, sp_size=nspix, ratio=ratio, map_ratio=map_ratio
    )

    abs_spix_indices = get_abs_indices(
        pre_label_map,
        artefacts,
        centers,
        lab_vect,
        nspix_ratio_map,
        nspix,
        filter_dist=True,
    )
    abs_spix_indices = (
        torch.from_numpy(abs_spix_indices)
        .repeat(batchsize, 1, 1)
        .reshape(-1)
        .to(device)
    )
    pre_label_map = torch.from_numpy(pre_label_map).repeat(batchsize, 1, 1, 1)

    abs_pix_indices = (
        torch.arange(n_pixel, device=device)[None, None]
        .repeat(batchsize, 9, 1)
        .reshape(-1)
    )
    abs_batch_indices = (
        torch.arange(batchsize, device=device)[:, None, None]
        .repeat(1, 9, n_pixel)
        .reshape(-1)
        .long()
    )

    abs_indices = torch.stack(
        [abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)

    return (
        pre_label_map,
        abs_indices,
        object_map,
        lab_vect,
        nspix_ratio_map,
        centers,
        n_sp_vect,
    )


def normalisation_min_max(tensor):
    mini = tensor.min()
    maxi = tensor.max()

    if mini == maxi:
        return tensor

    return (tensor - mini) / (maxi - mini)


def check_background_dino(
    tensor,
    mask=None,
    threshold_dino=0.6,
    threshold_background=0.1,
    arch_type="vit_small",
    patch_size=8,
):
    if mask is None:
        is_background = np.zeros((2), dtype=int)  # 0=artefacts, 1=objects
        return is_background
    model = dino.create_model(arch_type, patch_size)

    # reconversion to RGB
    tensor = torch.from_numpy(
        lab2rgb(tensor[0].permute(1, 2, 0).detach().cpu().numpy())
    ).permute(2, 0, 1)[None, :]

    tensor_norm = normalisation_min_max((tensor))
    attentions_map = dino.infere_one_image(
        model, threshold_dino, tensor_norm, patch_size
    )
    all_attention = sum(
        attentions_map[i] * 1 / attentions_map.shape[0]
        for i in range(attentions_map.shape[0])
    )
    all_attention = normalisation_min_max(all_attention)
    if all_attention.shape != mask.shape:
        all_attention = F.interpolate(
            all_attention.unsqueeze(0).unsqueeze(0), size=mask.shape, mode="nearest")[0, 0]

    label_map = mask.astype(int)
    if all_attention.shape != mask.shape:
        label_map = label_map[:-1, :-1]
    label_map = torch.Tensor(label_map)
    nb_object = int(label_map.max().item())
    is_background = np.zeros(nb_object + 1, dtype=float)
    for label in range(1, nb_object + 1):  # forget artefact
        coords = torch.zeros_like(label_map)
        coords[label_map == label] = 1
        if len(coords) == 0:
            continue
        is_background[label] = torch.sum(
            coords * all_attention) / torch.sum(coords)

    is_background = np.where(is_background <= threshold_background, 1, 0)

    return is_background, all_attention



def region_is_background(label_map):

    h, w = label_map.shape
    N = np.max(label_map)

    is_background = np.zeros(N + 1, dtype=int)

    for label in range(1, N + 1):  # skip artefacts

        coords = np.argwhere(label_map == label)
        if len(coords) == 0:
            continue
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        if min_x == 0 and max_x == w - 1:
            is_background[label] = 1
        elif min_y == 0 and max_y == h - 1:
            is_background[label] = 1

    return is_background


def get_mask_centroids(mask, n_sp):

    # Centroids (from scikit-image)
    coord = np.array(np.nonzero(mask), dtype=float).T
    rng = np.random.RandomState(123)
    idx_full = np.arange(len(coord), dtype=int)
    idx = np.sort(rng.choice(idx_full, min(n_sp, len(coord)), replace=False))
    # no subsampling here ?
    centroids, _ = kmeans2(coord, coord[idx], iter=5)

    # Closest centroids
    dist = cdist(coord, centroids)
    closest_centroids = np.argmin(dist, axis=1)

    return centroids, closest_centroids


def map_from_masks_ratio(masks, is_background, ind_to_one_hot, sp_size=None, ratio=2):

    h, w = masks.shape[1:3]
    if not sp_size:
        sp_size = np.sqrt(h * w)
    if np.any(np.sum(masks, axis=0) > 1):
        warnings.warn("regions overlap in masks")

    # Size object and background
    n_sp_max = 0
    Sb = 0
    So = 0
    for i, mask in enumerate(masks):
        area = mask.sum()
        if is_background[i + 1]:  # careful artefacts
            Sb += area
        else:
            So += area

    Nb = np.ceil(Sb * sp_size / (Sb + So))
    No = np.ceil(So * sp_size / (Sb + So))
    No_new = min(int(ratio * No), int(sp_size - Nb / 5))  # to let a few in B
    Nb_new = sp_size - No_new

    ratio_So = 1
    ratio_Sb = 1
    if No:
        ratio_So = No_new / No
    if Nb:
        ratio_Sb = Nb_new / Nb
    for i, mask in enumerate(masks):
        area = mask.sum()
        if is_background[i + 1]:  # careful artefacts
            n_sp = max(1, int(area / (Sb) * (Nb_new)))
        else:
            n_sp = max(1, int(int(area / (So) * (No_new))))
        n_sp_max += n_sp

    label_map = np.zeros((h, w), dtype=np.int64)
    centers = np.zeros((n_sp_max, 2), dtype=float)
    lab_vect = np.zeros((n_sp_max), dtype=np.int64)
    nspix_ratio_map = np.ones((h, w), dtype=float)
    n_sp_vect = np.zeros((n_sp_max), dtype=np.int64)
    curr_max_label = 0
    for i, mask in enumerate(masks):
        area = mask.sum()
        if is_background[i + 1]:  # careful artefacts
            n_sp = max(1, int(area / (Sb) * (Nb_new)))
            ratio_i = ratio_Sb
        else:
            n_sp = max(1, int(int(area / (So) * (No_new))))
            ratio_i = ratio_So

        if n_sp == 1:
            label_map[mask.astype(bool)] = curr_max_label
            nspix_ratio_map[mask.astype(bool)] = ratio_i
            positions = np.argwhere(mask == 1)
            centers[curr_max_label] = np.mean(positions, axis=0)
            lab_vect[curr_max_label] = ind_to_one_hot[i]
            n_sp_vect[curr_max_label] = 1
            curr_max_label += 1
        else:
            centers_i, closest_centers_i = get_mask_centroids(mask, n_sp)
            n_sp = centers_i.shape[0]
            centers[curr_max_label: curr_max_label + n_sp] = centers_i
            lab_vect[curr_max_label: curr_max_label + n_sp] = ind_to_one_hot[i]
            label_map[mask > 0] = closest_centers_i + curr_max_label
            nspix_ratio_map[mask > 0] = ratio_i
            n_sp_vect[curr_max_label: curr_max_label + n_sp] = n_sp
            curr_max_label += n_sp

    centers = np.array(centers)
    return label_map, centers, lab_vect, nspix_ratio_map, n_sp_vect


def map_from_masks_ratio2(
    masks,
    is_background,
    ind_to_one_hot,
    sp_size=None,
    ratio=2,
    map_ratio=None,  # NEW
):
    """
    If map_ratio is provided (HxW per-pixel weights), allocate an object-wise number
    of superpixels proportional to the sum of map_ratio within each object, then
    seed that many centroids per object. The internal 'foreground/background' ratio
    logic is skipped in that case.

    If map_ratio is None, keep the original behavior.
    """
    h, w = masks.shape[1:3]
    if not sp_size:
        sp_size = int(np.sqrt(h * w))

    # --- BRANCH 1: external ratio map provided -> compute per-object n_sp from it ---
    if map_ratio is not None:
        ext = np.asarray(map_ratio, dtype=float)
        if ext.shape != (h, w):
            raise ValueError(
                f"map_ratio must have shape {(h, w)}, got {ext.shape}")

        K = masks.shape[0]
        # Sum of weights per object
        obj_weights = np.zeros(K, dtype=float)
        union_mask = np.zeros((h, w), dtype=bool)
        for i, mask in enumerate(masks):
            m = mask.astype(bool)
            union_mask |= m
            obj_weights[i] = float(ext[m].sum())

        # Fallback if all-zero weights: use object areas
        if obj_weights.sum() <= 0:
            obj_weights = np.array([mask.sum() for mask in masks], dtype=float)
            if obj_weights.sum() <= 0:
                # degenerate case: nothing in masks
                obj_weights = np.ones(K, dtype=float)

        # Target total superpixels (int)
        S = int(sp_size)

        # Largest Remainder Method with min 1 sp per object when possible
        quotas = obj_weights / (obj_weights.sum() + 1e-12) * S
        base = np.floor(quotas).astype(int)

        # Ensure at least 1 per object (if S >= K). If S < K, we keep >=1 anyway
        # to match the existing code style, which might exceed S in that edge case.
        base = np.maximum(base, 1)
        total = int(base.sum())

        if total < S:
            # distribute remaining by largest fractional parts
            rem = quotas - np.floor(quotas)
            order = np.argsort(-rem)
            need = S - total
            for idx in order:
                if need == 0:
                    break
                base[idx] += 1
                need -= 1
            total = int(base.sum())
        elif total > S:
            # try removing from the smallest fractional parts, but never below 1
            rem = quotas - np.floor(quotas)
            order = np.argsort(rem)  # ascending
            excess = total - S
            for idx in order:
                if excess == 0:
                    break
                if base[idx] > 1:
                    base[idx] -= 1
                    excess -= 1
            # If still excess > 0, we keep > S (edge case S < K), same as old behavior.

        # Prepare outputs
        n_sp_per_obj = base
        n_sp_max = int(n_sp_per_obj.sum())

        label_map = np.zeros((h, w), dtype=np.int64)
        centers = np.zeros((n_sp_max, 2), dtype=float)
        lab_vect = np.zeros((n_sp_max), dtype=np.int64)
        nspix_ratio_map = np.ones((h, w), dtype=float)
        n_sp_vect = np.zeros((n_sp_max), dtype=np.int64)

        # Normalize ext on the union of objects so that mean == 1 (keeps budgets stable)
        mean_ext = float(ext[union_mask].mean()) if union_mask.any() else 1.0
        if mean_ext > 0:
            ext_norm = ext / mean_ext
        else:
            ext_norm = np.ones_like(ext, dtype=float)

        curr = 0
        for i, mask in enumerate(masks):
            m = mask.astype(bool)
            n_sp = int(n_sp_per_obj[i])

            if n_sp <= 1:
                # Single region
                label_map[m] = curr
                nspix_ratio_map[m] = ext_norm[m]
                pos = np.argwhere(m)
                centers[curr] = np.mean(
                    pos, axis=0) if pos.size else np.array([0, 0])
                lab_vect[curr] = ind_to_one_hot[i]
                n_sp_vect[curr] = 1
                curr += 1
            else:
                # K-means seeding inside the object
                centers_i, closest = get_mask_centroids(mask, n_sp)
                n_sp_i = centers_i.shape[0]
                centers[curr: curr + n_sp_i] = centers_i
                lab_vect[curr: curr + n_sp_i] = ind_to_one_hot[i]
                label_map[m] = closest + curr
                nspix_ratio_map[m] = ext_norm[m]
                n_sp_vect[curr: curr + n_sp_i] = n_sp_i
                curr += n_sp_i

        centers = np.array(centers)
        return label_map, centers, lab_vect, nspix_ratio_map, n_sp_vect

    # --- BRANCH 2: original behavior (no external map), unchanged ---
    if np.any(np.sum(masks, axis=0) > 1):
        warnings.warn("regions overlap in masks")

    # Size object and background
    n_sp_max = 0
    Sb = 0
    So = 0
    for i, mask in enumerate(masks):
        area = mask.sum()
        # careful artefacts (indexing follows existing code)
        if is_background[i + 1]:
            Sb += area
        else:
            So += area

    Nb = np.ceil(Sb * sp_size / (Sb + So)) if (Sb + So) else 0
    No = np.ceil(So * sp_size / (Sb + So)) if (Sb + So) else 0
    No_new = min(int(ratio * No), int(sp_size - Nb / 5)
                 ) if No else 0  # leave few in B
    Nb_new = sp_size - No_new

    ratio_So = No_new / No if No else 1
    ratio_Sb = Nb_new / Nb if Nb else 1

    for i, mask in enumerate(masks):
        area = mask.sum()
        if is_background[i + 1]:
            n_sp = max(1, int(area / (Sb) * (Nb_new))) if Sb else 1
        else:
            n_sp = max(1, int(area / (So) * (No_new))) if So else 1
        n_sp_max += n_sp

    label_map = np.zeros((h, w), dtype=np.int64)
    centers = np.zeros((n_sp_max, 2), dtype=float)
    lab_vect = np.zeros((n_sp_max), dtype=np.int64)
    nspix_ratio_map = np.ones((h, w), dtype=float)
    n_sp_vect = np.zeros((n_sp_max), dtype=np.int64)

    curr_max_label = 0
    for i, mask in enumerate(masks):
        area = mask.sum()
        if is_background[i + 1]:
            n_sp = max(1, int(area / (Sb) * (Nb_new))) if Sb else 1
            ratio_i = ratio_Sb
        else:
            n_sp = max(1, int(area / (So) * (No_new))) if So else 1
            ratio_i = ratio_So

        if n_sp == 1:
            label_map[mask.astype(bool)] = curr_max_label
            nspix_ratio_map[mask.astype(bool)] = ratio_i
            positions = np.argwhere(mask == 1)
            centers[curr_max_label] = np.mean(positions, axis=0)
            lab_vect[curr_max_label] = ind_to_one_hot[i]
            n_sp_vect[curr_max_label] = 1
            curr_max_label += 1
        else:
            centers_i, closest_centers_i = get_mask_centroids(mask, n_sp)
            n_sp = centers_i.shape[0]
            centers[curr_max_label: curr_max_label + n_sp] = centers_i
            lab_vect[curr_max_label: curr_max_label + n_sp] = ind_to_one_hot[i]
            label_map[mask > 0] = closest_centers_i + curr_max_label
            nspix_ratio_map[mask > 0] = ratio_i
            n_sp_vect[curr_max_label: curr_max_label + n_sp] = n_sp
            curr_max_label += n_sp

    centers = np.array(centers)
    return label_map, centers, lab_vect, nspix_ratio_map, n_sp_vect


def get_abs_indices(
    pre_label_map,
    artefacts,
    centers,
    lab_vect,
    nspix_ratio_map,
    nspix,
    filter_dist=False,
    ratio_init=1.65,
    artefacts_mode="late",
):

    h, w = pre_label_map.shape

    # Matrix distance
    dist_matrix = cdist(centers, centers)
    # Filtering distance inside same object
    same_object = lab_vect[:, None] == lab_vect[None, :]
    dist_matrix[~same_object] = np.inf

    if filter_dist:
        nspix_ratio_seeds = nspix_ratio_map[
            (centers[:, 0]).astype("int"), (centers[:, 1]).astype("int")
        ]
        th = ratio_init * np.sqrt(h * w / nspix)
        dist_matrix[(dist_matrix * nspix_ratio_seeds) > th] = np.inf

    # Get 9 nearest SP inside object
    relative_sp_indices = np.argsort(dist_matrix, axis=1)[:, :9]
    # Filter valid neighbors
    valid_number = np.minimum(9, np.sum(dist_matrix != np.inf, axis=1))
    for i in range(relative_sp_indices.shape[0]):
        relative_sp_indices[i, valid_number[i]:] = i

    # Pixel Association
    h, w = pre_label_map.shape
    abs_spix_indices = relative_sp_indices[pre_label_map.reshape(1, h * w)].transpose(
        0, 2, 1
    )

    if artefacts_mode == "late":
        coord = np.array(np.nonzero(artefacts), dtype=float).T
        if coord.shape[0]:
            dist_art = cdist(coord, centers)

            # Get 9 nearest SP
            relative_sp_indices_art = np.argsort(dist_art, axis=1)[:, :9]
            indices_artefacts = np.argwhere(artefacts)
            indices_plats = indices_artefacts[:,
                                              1] + indices_artefacts[:, 0] * w
            abs_spix_indices[0, :, indices_plats] = relative_sp_indices_art[
                : len(indices_plats), :
            ]
            # affect artefacts to closest SP
            for idx, (y, x) in enumerate(np.argwhere(artefacts)):
                pre_label_map[y, x] = relative_sp_indices_art[idx, 0]

    abs_spix_indices = abs_spix_indices.reshape(-1)  # .long()
    return abs_spix_indices


def labelmap2onehot(label_map, background=None, channels_last=False):
    h, w = label_map.shape
    labels = np.unique(label_map)
    if background is not None:
        labels = labels[labels != background]
    n_labels = len(labels)
    ind_to_one_hot = np.zeros((n_labels))
    onehot = np.zeros((n_labels, h, w), dtype=np.uint8)
    for l, label in enumerate(labels):
        onehot[l, :, :] = label_map == label
        ind_to_one_hot[l] = label
    if channels_last:
        onehot = onehot.transpose(1, 2, 0)
    return onehot, ind_to_one_hot


class Meter:
    def __init__(self, ema_coef=0.9):
        self.ema_coef = ema_coef
        self.params = {}

    def add(self, params: dict, ignores: list = []):
        for k, v in params.items():
            if k in ignores:
                continue
            if not k in self.params.keys():
                self.params[k] = v
            else:
                self.params[k] -= (1 - self.ema_coef) * (self.params[k] - v)

    def state(self, header="", footer=""):
        state = header
        for k, v in self.params.items():
            state += f" {k} {v:.6g} |"
        return state + " " + footer

    def reset(self):
        self.params = {}


def naive_sparse_bmm(sparse_mat, dense_mat, transpose=False):
    if transpose:
        return torch.stack(
            [
                torch.sparse.mm(s_mat, d_mat.t())
                for s_mat, d_mat in zip(sparse_mat, dense_mat)
            ],
            0,
        )
    else:
        return torch.stack(
            [
                torch.sparse.mm(s_mat, d_mat)
                for s_mat, d_mat in zip(sparse_mat, dense_mat)
            ],
            0,
        )


def sparse_permute(sparse_mat, order):
    values = sparse_mat.coalesce().values()
    indices = sparse_mat.coalesce().indices()
    indices = torch.stack([indices[o] for o in order], 0).contiguous()
    return torch.sparse_coo_tensor(indices, values)


def plot_histogram_feature(tensor_feature, batch, channel):

    data = tensor_feature[batch, channel].flatten().cpu().detach().numpy()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, color="blue", edgecolor="black")
    plt.title(f"Histogram for Batch {batch}, Channel {channel}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


def save_histogram_feature(tensor_feature, batch, path_to_save):
    _, c, _, _ = tensor_feature.size()

    for channel in range(c):

        image_data = tensor_feature[batch, channel].cpu().detach().numpy()
        histogram_data = image_data.flatten()

        # Create the subplots
        _, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the image
        axes[0].imshow(image_data, cmap="gray")
        axes[0].set_title(f"Image for Batch {batch}, Channel {channel}")
        axes[0].axis("off")

        # Plot the histogram
        axes[1].hist(histogram_data, bins=30, color="blue", edgecolor="black")
        axes[1].set_title("Histogram of Pixel Intensities")
        axes[1].set_xlabel("Pixel Intensity")
        axes[1].set_ylabel("Frequency")

        # Adjust layout
        plt.tight_layout()

        # Save the plot if a save_path is provided
        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)
            name_file = os.path.join(path_to_save, f"feature_{channel}.png")
            plt.savefig(name_file)
            print(f"Plot saved to {name_file}")


def plot_histogram_weight(model, nb_bins=50):
    weights = []

    for name, param in model.named_parameters():
        if "weight" in name:
            weights.extend(param.data.cpu().numpy().flatten())

    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=nb_bins, color="blue", alpha=0.7)
    plt.title("Histogramme des poids du modèle")
    plt.xlabel("Valeur des poids")
    plt.ylabel("Fréquence")
    plt.show()
