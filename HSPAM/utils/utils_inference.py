import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab
from skimage.segmentation import mark_boundaries
import pandas as pd

import ssn.enforce_connectivity_mask
from ssn._enforce_connectivity_mask import _enforce_connectivity_labels_with_mask
from skimage.segmentation._slic import _enforce_label_connectivity_cython
import matplotlib.colors as mcolors

from ssn.model import create_model
from ssn.utils import (
    precompute_abs_indices,
    compute_inputs,
)
import cv2
import os
import torch



def interactive_attention_map(image, object_map, alpha_overlay=0.5, save_path=None):

    if object_map is None:
        raise ValueError(
            "interactive_attention_map requires a non-null object_map.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must be RGB (H, W, 3) for interactive mode.")

    if object_map.shape != image.shape[:2]:
        object_map_resized = cv2.resize(
            object_map.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        object_map_resized = object_map

    H, W = object_map_resized.shape
    attention = np.zeros((H, W), dtype=np.float32)

    att_values = {}
    for lab in np.unique(object_map_resized):
        if lab in (0, -1):
            continue
        att_values[int(lab)] = 1.0
        attention[object_map_resized == lab] = 1.0

    fig, (ax_img, ax_att) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        "Interactive attention mode\n"
        "Left click = low attention, right click = high attention, "
        "press 'q' or 'enter' to validate",
        fontsize=10,
    )

    overlay = overlay_object_map(
        image, object_map_resized, alpha=alpha_overlay)
    img_handle = ax_img.imshow(overlay)
    ax_img.set_title("Image + objects SAM")
    ax_img.axis("off")

    scatter_handle = ax_img.scatter(
        [], [], c=[], s=40, marker="+", linewidths=2
    )
    att_handle = ax_att.imshow(
        attention, cmap="viridis"
    )
    ax_att.set_title("User attention map")
    ax_att.axis("off")

    state = {
        "attention": attention,
        "object_map": object_map_resized,
        "overlay": overlay,
        "img_handle": img_handle,
        "att_handle": att_handle,
        "ax_att": ax_att,
        "att_values": att_values,
        "scatter_handle": scatter_handle,
        "click_xs": [],
        "click_ys": [],
        "click_colors": [],
    }

    def on_click(event):
        if event.inaxes not in (ax_img, ax_att):
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or x >= W or y < 0 or y >= H:
            return

        label = int(state["object_map"][y, x])
        if label in (0, -1):
            return

        att_values = state["att_values"]
        current = att_values.get(label, 1.0)

        if event.button == 1:
            new_val = current / 2.0
        elif event.button == 3:
            new_val = current * 2.0
        else:
            return

        att_values[label] = new_val

        mask = state["object_map"] == label
        state["attention"][mask] = float(new_val)

        att = state["attention"]
        state["att_handle"].set_data(att)
        vmin = float(att.min())
        vmax = float(att.max())
        if vmax == vmin:
            vmax = vmin + 1.0
        state["att_handle"].set_clim(vmin=vmin, vmax=vmax)

        obj_map = state["object_map"]
        if event.button == 1:
            color = (1.0, 0.0, 0.0)  # red
        elif event.button == 3:
            color = (0.0, 0.0, 1.0)  # blue
        else:
            color = (0.0, 0.0, 0.0)

        state["click_xs"].append(event.xdata)
        state["click_ys"].append(event.ydata)
        state["click_colors"].append(color)

        scat = state["scatter_handle"]
        xs = np.array(state["click_xs"], dtype=np.float32)
        ys = np.array(state["click_ys"], dtype=np.float32)
        colors = np.array(state["click_colors"], dtype=np.float32)
        offsets = np.column_stack([xs, ys])
        scat.set_offsets(offsets)
        scat.set_color(colors)

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("q", "enter"):
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        overlay_img = state["overlay"].copy()
        H, W, _ = overlay_img.shape

        xs = np.array(state["click_xs"], dtype=np.int32)
        ys = np.array(state["click_ys"], dtype=np.int32)
        colors = np.array(state["click_colors"], dtype=np.float32)

        size = 5
        for (x, y, col) in zip(xs, ys, colors):
            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            r = int(col[0] * 255)
            g = int(col[1] * 255)
            b = int(col[2] * 255)
            y_min = max(0, y - size)
            y_max = min(H - 1, y + size)
            overlay_img[y_min:y_max + 1, x] = (r, g, b)
            x_min = max(0, x - size)
            x_max = min(W - 1, x + size)
            overlay_img[y, x_min:x_max + 1] = (r, g, b)

        plt.imsave(save_path, overlay_img)
    return state["attention"].astype(np.float32)



def achievable_segmentation_accuracy(superpixel, label):
    flat_gt = label.ravel()
    flat_S = superpixel.ravel()

    unique_gt = np.unique(flat_gt)
    unique_S = np.unique(flat_S)

    label_to_idx_gt = {label: idx for idx, label in enumerate(unique_gt)}
    label_to_idx_S = {label: idx for idx, label in enumerate(unique_S)}

    confusion = np.zeros((len(unique_S), len(unique_gt)), dtype=np.int64)

    for s_label, gt_label in zip(flat_S, flat_gt):
        i = label_to_idx_S[s_label]
        j = label_to_idx_gt[gt_label]
        confusion[i, j] += 1

    max_overlaps = np.max(confusion, axis=1)
    return np.sum(max_overlaps) / flat_S.size


def visualize_asa_errors(
    image, superpixel, label, img_sp, final_object_map, alpha=0.5
):

    if len(image.shape) == 4:  # e.g., (1, H, W, 3)
        image = image[0]
    elif len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # Scale to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    overlay = image.copy()

    error_mask = np.zeros_like(superpixel, dtype=bool)
    unique_id = np.unique(superpixel)
    nb_label = np.unique(label).shape[0]

    for sp_id in unique_id:
        sp_mask = superpixel == sp_id
        label_sp = np.copy(label)
        label_sp[sp_mask == 0] = -1
        label_hist = np.histogram(label_sp, bins=range(nb_label + 1))
        maximum_regionsize = label_hist[0].argmax()
        labels_in_sp = label_sp != maximum_regionsize
        error_mask[sp_mask & labels_in_sp] = True

    red_overlay = np.zeros_like(image)
    red_overlay[error_mask] = np.array([1, 0, 0])
    result = cv2.addWeighted(img_sp, 1, red_overlay, alpha, 0)
    result = result - result.min()
    result = result / (result.max() - result.min())
    asa = 1 - error_mask.sum() / error_mask.size

    # Ensure non-interactive plotting
    plt.ioff()
    fig = plt.figure(figsize=(10, 5))

    plt.subplot(231)
    plt.imshow(image)
    plt.title("Input image")
    plt.axis("off")

    plt.subplot(232)
    plt.imshow(label)
    plt.title("Label")
    plt.axis("off")

    plt.subplot(233)
    plt.imshow(result)
    plt.title("result with ASA errors (red)")
    plt.axis("off")

    plt.subplot(234)
    plt.imshow(img_sp)
    plt.title("Superpixels")
    plt.axis("off")

    plt.subplot(235)
    plt.imshow(final_object_map)
    plt.title("final_object_map")
    plt.axis("off")

    plt.subplot(236)
    plt.imshow(red_overlay)
    plt.title("ASA errors (red)")
    plt.axis("off")

    plt.tight_layout()
    return fig


def _ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def default_on_result_saver(out_folder, pos_scale, color_scale, override_results=False):
    def on_result(res):
        img_name = res["img_name"]
        nsp = res["nsp"]
        label = res["label"]
        final_object_map = res["final_object_map"]
        fig = res["fig"]
        path_img = res["path_img"]

        output_path = f"{out_folder}/{nsp}"
        dir_path_label = f"{output_path}/label/"
        dir_path_img = f"{output_path}/img/"
        dir_path_sam = f"{output_path}/sam_map/"
        dir_path_error_asa = f"{output_path}/error_asa/"
        _ensure_dirs([dir_path_label, dir_path_img,
                     dir_path_sam, dir_path_error_asa])

        res_dest_label = f"{dir_path_label}{img_name}_{nsp}_label.npy"
        res_dest_sam_map_img = f"{dir_path_sam}{img_name}_sam_final_map_2.png"
        res_dest_error_asa = f"{dir_path_error_asa}{img_name}_error_asa.png"
        res_dest_img = f"{dir_path_img}{img_name}_{pos_scale}_{color_scale}.png"

        if override_results or (not os.path.exists(res_dest_label)):
            np.save(res_dest_label, label)
            image = plt.imread(path_img)
            plt.imsave(res_dest_img, mark_boundaries(image, label))
            plt.imsave(res_dest_sam_map_img, final_object_map)
            fig.savefig(res_dest_error_asa)
    return on_result


def fill_asa_dataframe(asa_dict, image_name, nsp, asa, nsp_range):
    """Update ASA dataframe for a given image and nsp value."""
    image_name = image_name + ".png"
    exist = (asa_dict["image"] == image_name).any()
    if not exist:
        asa_dict.loc[len(asa_dict)] = [image_name] + [None for _ in nsp_range]
    asa_dict.loc[asa_dict["image"] == image_name, str(nsp)] = asa
    return asa_dict
def keep_largest_objects_by_label(mask, nspix):
    # Keep nspix largest non-zero labels and relabel them to 1..k (0 preserved)
    if mask is None:
        return mask
    if mask.ndim != 2:
        raise ValueError("mask must be [H, W]")
    if nspix is None:
        raise ValueError("nspix must be provided")
    if nspix <= 0:
        return torch.zeros_like(mask) if isinstance(mask, torch.Tensor) else np.zeros_like(mask)

    # Torch path
    if isinstance(mask, torch.Tensor):
        idx = mask.to(torch.long).reshape(-1)
        if idx.numel() == 0:
            return mask
        max_label = int(idx.max().item())
        if max_label == 0:
            return mask  # only artifacts
        counts = torch.bincount(idx, minlength=max_label + 1)
        counts[0] = 0
        labels_nz = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
        num_objs = int(labels_nz.numel())
        k = min(int(nspix), num_objs)
        if num_objs == 0 or k == 0:
            return torch.zeros_like(mask)
        order = torch.argsort(counts[labels_nz], descending=True)
        keep_labels = labels_nz[order[:k]]

        lut = torch.zeros(max_label + 1, dtype=mask.dtype, device=mask.device)
        new_vals = torch.arange(1, k + 1, device=mask.device).to(mask.dtype)
        lut[keep_labels] = new_vals
        out = lut[idx].view_as(mask)
        return out

    # NumPy path
    if isinstance(mask, np.ndarray):
        flat = mask.reshape(-1).astype(np.int64, copy=False)
        if flat.size == 0:
            return mask
        max_label = int(flat.max())
        if max_label == 0:
            return mask
        counts = np.bincount(flat, minlength=max_label + 1)
        counts[0] = 0
        labels_nz = np.flatnonzero(counts > 0)
        num_objs = int(labels_nz.size)
        k = min(int(nspix), num_objs)
        if num_objs == 0 or k == 0:
            return np.zeros_like(mask)
        order = np.argsort(-counts[labels_nz])
        keep_labels = labels_nz[order[:k]]

        lut = np.zeros(max_label + 1, dtype=mask.dtype)
        lut[keep_labels] = np.arange(1, k + 1, dtype=mask.dtype)
        out = lut[flat].reshape(mask.shape)
        return out

    raise TypeError("mask must be a torch.Tensor or numpy.ndarray")

def overlay_object_map(
    image,
    object_map,
    alpha=0.5,
    seed=153,
    background_labels=(0, -1),
):
    """
    Overlay the object map (object_map) on the input image.

    Args:
        image (np.ndarray): image RGB (H, W, 3), dtype uint8 or float.
        object_map (np.ndarray): object map (H, W) or (H, W, 1), index values.
        alpha (float): weight of the object map in the fusion (0 = original image).
        seed (int | None): seed to generate reproducible colors.
        background_labels (tuple[int, ...]): labels to leave transparent (0 and/or -1 by default).

    Returns:
        np.ndarray: image RGB (uint8) avec overlay.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be in RGB format (H, W, 3).")

    if object_map is None:
        raise ValueError("object_map must not be None.")

    if object_map.ndim == 3 and object_map.shape[2] == 1:
        object_map = object_map[..., 0]

    if object_map.shape != image.shape[:2]:
        object_map = cv2.resize(
            object_map.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    object_map = object_map.astype(np.int64, copy=False)
    unique_labels, inv = np.unique(object_map, return_inverse=True)
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 256, size=(unique_labels.size, 3), dtype=np.uint8)
    for bg_label in background_labels:
        if bg_label in unique_labels:
            idx = np.where(unique_labels == bg_label)[0][0]
            colors[idx] = 0

    colored_map = colors[inv].reshape(image.shape[0], image.shape[1], 3)

    if image.dtype != np.uint8:
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image

    overlay = image_uint8.astype(np.float32) * (1 - alpha) + \
        colored_map.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay
