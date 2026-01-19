import warnings
import numpy as np
import skimage.measure
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
import operator


def get_sp_centers(merged_objects_label_map, n_sp):

    h, w = merged_objects_label_map.shape

    centroids = []
    centroids_labels = []
    for i in range(np.max(merged_objects_label_map) + 1):
        mask = merged_objects_label_map == i
        area = mask.sum()
        n_sp_i = max(1, int(n_sp * (area / (h * w))))
        centroids_i, steps = skimage.segmentation.slic_superpixels._get_mask_centroids(
            mask=mask, n_centroids=n_sp_i, multichannel=False
        )
        centroids.append(centroids_i)
        centroids_labels.append([i for _ in range(centroids_i.shape[0])])

    centroids_array = np.empty((0, 2))
    for centroids_i in centroids:
        centroids_array = np.concatenate(
            (centroids_array, centroids_i), axis=0)
    centroids_array = np.array(centroids_array)

    centroids_labels_array = np.empty((0))
    for centroids_labels_i in centroids_labels:
        centroids_labels_array = np.concatenate(
            (centroids_labels_array, centroids_labels_i), axis=0
        )
    centroids_labels_array = np.array(centroids_labels_array)

    return centroids_array, centroids_labels_array


def mask_slic_objects(image, masks, sp_size=None, compactness=10.0, relabel=True):
    h, w = image.shape[:2]
    if not sp_size:
        sp_size = np.sqrt(h * w)
    if np.any(np.sum(masks, axis=0) > 1):
        warnings.warn("regions overlap in masks")
    label_map = np.zeros((h, w), dtype=np.int64)
    curr_max_label = 0
    for i, mask in enumerate(masks):
        area = mask.sum()
        n_sp = max(1, int(sp_size * (area / (h * w))))
        if n_sp == 1:
            label_map[mask.astype(bool)] = 1 + curr_max_label
            curr_max_label += 1
        else:
            slic = skimage.segmentation.slic(
                image, mask=mask, n_segments=n_sp, compactness=compactness
            )
            mask_slic = slic > 0
            label_map[mask_slic] = slic[mask_slic] + curr_max_label
            curr_max_label += np.max(slic)
    if relabel:
        label_map, forward_map, inverse_map = skimage.segmentation.relabel_sequential(
            label_map
        )
    return label_map


def merge_artefacts_from_sp(image, objects_label_map, sp_label_map, win_s=4):
    """
    Merge artefact pixels into the object label of their most similar close
    superpixel.
    Artefact pixels are assumed to be label 0.
    Close superpixels are the ones in the (2win_s+1)^2 window.
    Similarity is computed as the distance to the mean color of the superpixel.
    """
    h, w = image.shape[:2]
    unique_sp_labels = np.unique(sp_label_map)
    unique_objects_labels = np.unique(objects_label_map)

    sp_color_features = {}
    for label in unique_sp_labels:
        if label == 0:  # assuming 0 is for artefacts
            continue
        mean_color = image[sp_label_map == label].mean(axis=0)
        sp_color_features[label] = mean_color

    merged_objects_label_map = objects_label_map.copy()

    ii, jj = np.where(sp_label_map == 0)
    for i, j in zip(ii, jj):
        # Find most similar SP in window
        ia, ib = max(0, i - win_s), min(h, i + win_s) + 1
        ja, jb = max(0, j - win_s), min(w, j + win_s) + 1
        window = sp_label_map[ia:ib, ja:jb]
        window_labels = np.unique(window)
        window_labels = window_labels[window_labels != 0]
        if len(window_labels) == 0:
            # No other label found in window, assign random label
            merged_objects_label_map[i, j] = np.random.choice(
                unique_objects_labels[1:])
        else:
            dists = np.array(
                [
                    np.mean((image[i, j] - sp_color_features[wl]) ** 2)
                    for wl in window_labels
                ]
            )
            argmin_label = window_labels[np.argmin(dists)]
            # Assign main object to artefacts
            ol = objects_label_map[sp_label_map == argmin_label]
            if len(np.unique(ol)) != 1:
                raise ValueError(
                    f"SP {argmin_label} is not fully contained in an object."
                )
            merged_objects_label_map[i, j] = ol[0]

    if np.any(merged_objects_label_map) == 0:
        raise ValueError(f"Artefact pixels remaining.")

    return merged_objects_label_map


def mask_local_nearest_neighbor(mask_slic_label_map, image, artefacts, win_s=4):
    """Algorithmic implementation of kmeans clustering.
    Pixels at 1 in the artefacts map are associated to closest superpixel in
    mask_slic_label_map in a (2win_s+1)^2 window.
    If only artefact pixels in the neighborhood, pixel receive a random label."""

    h, w = image.shape[:2]

    unique_labels = np.unique(mask_slic_label_map)
    max_lab = np.max(mask_slic_label_map)

    mask_slic_label_map_copy = np.copy(mask_slic_label_map)

    avg_features = {}
    for label in unique_labels:
        mask = mask_slic_label_map == label
        mean_color = image[mask].astype("float").mean(axis=0)
        avg_features[label] = mean_color

    max_lab = np.max(mask_slic_label_map)
    mask_slic_label_map_copy[artefacts == 1] = np.random.choice(
        unique_labels + max_lab + 1, np.sum(artefacts)
    )
    for i in range(0, h):
        for j in range(0, w):
            if artefacts[i, j]:
                win = mask_slic_label_map[
                    max(0, i - win_s): min(h, i + win_s) + 1,
                    max(0, j - win_s): min(w, j + win_s) + 1,
                ]
                lab_win = np.unique(win)
                min_dist = 9999999
                lab = mask_slic_label_map[i, j]
                for l in lab_win:
                    if l == 0:
                        continue
                    if l > max_lab:
                        continue
                    dist = np.mean(
                        (image[i, j].astype("float") - avg_features[l]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        lab = l
                        artefacts[i, j] = 0
                mask_slic_label_map_copy[i, j] = lab

    return mask_slic_label_map_copy, artefacts


def labelmap2onehot(label_map, background=None, channels_last=False):
    h, w = label_map.shape
    labels = np.unique(label_map)
    if background is not None:
        labels = labels[labels != background]
    n_labels = len(labels)
    onehot = np.zeros((n_labels, h, w), dtype=np.uint8)
    for l, label in enumerate(labels):
        onehot[l, :, :] = label_map == label
    if channels_last:
        onehot = onehot.transpose(1, 2, 0)
    return onehot


def onehot2labelmap(onehot, channels_last=False, start_label=1, unlabeled_label=-1):
    channels_axis = -1 if channels_last else 0
    label_map = np.argmax(onehot, axis=channels_axis)
    unlabeled = np.all(onehot == 0, axis=0)
    if np.any(unlabeled):
        # warnings.warn('unlabeled pixels found')
        label_map[unlabeled] = unlabeled_label
    label_map[label_map != unlabeled_label] += start_label
    return label_map


def sammasks2onehot(sam_masks, sort_by_area=True, return_unlabeled=False):
    if sort_by_area:
        sam_masks = sorted(
            sam_masks, key=operator.itemgetter("area"), reverse=True)
    onehot = np.stack([m["segmentation"] for m in sam_masks]).astype(np.uint8)
    if return_unlabeled:
        unlabeled = np.all(onehot == 0, axis=0).astype(np.uint8)
        return onehot, unlabeled
    else:
        return onehot


def center_points(onehot_segm, normalized=False):
    """Return the center point of each region in a segmentation.
    The input segmentation should be in channels-first onehot format.
    Center points are listed as (i, j) coordinates.
    The center point is computed as the maximum value of its distance transform.
    Padding is added to handle regions touching the image boundaries."""
    n_labels = onehot_segm.shape[0]
    centers = np.zeros((n_labels, 2))
    for l in range(n_labels):
        region = onehot_segm[l, :, :]
        region = np.pad(region, 1, mode="constant", constant_values=0)
        dist_transform = distance_transform_edt(region)
        centers[l] = np.unravel_index(
            np.argmax(dist_transform.reshape(-1)), dist_transform.shape
        )
    if normalized:
        centers /= onehot_segm.shape[1:]
    return centers


def filter_objects(onehot_segm, area_threshold):
    n_labels = onehot_segm.shape[0]
    areas = np.sum(onehot_segm.reshape(n_labels, -1), axis=1)
    mask = areas > area_threshold
    return mask


def filter_objects_cc(mask, area_threshold, background=0):
    # Label connected components and convert to onehot
    connected_components = skimage.measure.label(
        mask, background=background, connectivity=1
    )
    cc_onehot = labelmap2onehot(connected_components, background=background)
    # Filter small objects
    objects_mask = filter_objects(cc_onehot, area_threshold)
    return cc_onehot[objects_mask], cc_onehot[~objects_mask]


def reorder_label_map(label_map):

    unique_labels = np.unique(label_map)
    label_mapping = {
        old_label: new_label for new_label, old_label in enumerate(unique_labels)
    }
    new_label_map = np.copy(label_map)
    for old_label, new_label in label_mapping.items():
        new_label_map[label_map == old_label] = new_label

    return new_label_map
