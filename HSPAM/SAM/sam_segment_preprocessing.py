import os
import time
import numpy as np
import skimage.segmentation
import skimage.morphology as morph
from skimage.io import imsave
import matplotlib.pyplot as plt
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch.nn.functional as F

from SAM import utils
from ultralytics import FastSAM

# sam_model_path = './SAM/sam_vit_h_4b8939.pth'
sam_model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sam_vit_h_4b8939.pth"
)
VERBOSE = False

SAVE_INIT = False
save_path = "tmp"


def plot_grid(image, num_points):
    h, w, _ = image.shape
    x_spacing = w / (num_points + 1)
    y_spacing = h / (num_points + 1)

    x_coords = np.linspace(x_spacing, w - x_spacing, num_points)
    y_coords = np.linspace(y_spacing, h - y_spacing, num_points)

    plt.imshow(image)
    plt.axis("off")
    for x in x_coords:
        for y in y_coords:
            plt.plot(x, y, "o", color=[0, 0, 1], markersize=4)


def show_objects_with_background(
    object_map,
    image_rgb,
    alpha=0.25,
    color_artefacts=[1, 1, 0],
    sp_color=[0, 0, 0],
    superpixels=None,
):

    plt.figure(666)
    img = np.copy(image_rgb).astype(np.float32) / 255

    colored_mask = np.zeros_like(image_rgb, dtype=np.float32)
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


def segment_with_sam(
    sam_model, image, area_threshold, points_per_side, point_grids=None
):

    image_i = image  # torch.from_numpy(image).cuda()

    tic = time.time()
    if points_per_side is not None:

        mask_generator = SamAutomaticMaskGenerator(
            sam_model,
            points_per_side=points_per_side,
            min_mask_region_area=area_threshold,
        )

    elif point_grids is not None:
        mask_generator = SamAutomaticMaskGenerator(
            sam_model,
            points_per_side=None,
            point_grids=point_grids,
            min_mask_region_area=area_threshold,
        )
    masks = mask_generator.generate(image_i)

    if VERBOSE:
        toc = time.time()
        print(f"Time to SAM: {toc - tic:.2f} s")

    return masks


def process_segmentation(image, sam_masks, area_threshold, selem, is_not_sam=False):

    # Filter main objects and unlabeled regions
    if is_not_sam:
        sam_onehot = np.array(sam_masks[0])
        unlabeled = np.all(sam_onehot == 0, axis=0).astype(np.uint8)
    else:
        sam_onehot, unlabeled = utils.sammasks2onehot(
            sam_masks, return_unlabeled=True)
    filter_objects_mask = utils.filter_objects(
        sam_onehot, area_threshold=area_threshold
    )
    main_objects = sam_onehot[filter_objects_mask]
    main_objects_union = np.any(main_objects, axis=0)
    unlabeled = ~main_objects_union
    # fully labeled and no intersection
    assert np.all(main_objects_union != unlabeled)

    # Greedily remove overlapping of unlabeled and objects sorted by increasing area
    bg_objects = np.vstack((unlabeled[np.newaxis], main_objects[::-1]))
    greedy_label_map = np.argmax(bg_objects, axis=0)
    greedy_objects = utils.labelmap2onehot(greedy_label_map)

    # Filter background and main objects into artefacts
    filtered_objects = []
    artefacts = np.zeros(image.shape[:2], dtype=np.uint8)
    for obj in greedy_objects:
        # Filter object: object and artefacts by morphology
        obj_filtered = morph.opening(
            obj, footprint=selem) * 1  # , mode='constant'
        artefacts += (obj - obj_filtered) * 1
        if np.all(obj_filtered == 0):
            continue
        # Filter object: CC and area threshold
        objs_cc, objs_artefact = utils.filter_objects_cc(
            obj_filtered, area_threshold)
        if len(objs_cc) > 0:
            filtered_objects.append(objs_cc)  # keep different main CC
        if len(objs_artefact) > 0:
            # merge CC into artefacts map
            artefacts += objs_artefact.sum(axis=0)
    if len(filtered_objects) > 0:
        filtered_objects = [np.vstack([n for n in filtered_objects])]

    # Filter artefacts: morpho, CC and area threshold
    if np.any(artefacts != 0):
        objs_artefacts = (
            morph.opening(artefacts, footprint=selem) * 1
        )  # , mode='constant'
        artefacts_filtered = artefacts - objs_artefacts
        if np.any(objs_artefacts != 0):
            objs_cc, objs_artefact = utils.filter_objects_cc(
                objs_artefacts, area_threshold
            )
            if len(objs_cc) > 0:
                filtered_objects.append(objs_cc)
            if len(objs_artefact) > 0:
                artefacts_filtered += objs_artefact.sum(axis=0)
    else:
        artefacts_filtered = artefacts

    filtered_objects = np.vstack([n for n in filtered_objects])
    n_objects = filtered_objects.shape[0]
    if n_objects == 0:
        print("No objects found after processing.")
        return

    label_map = utils.onehot2labelmap(filtered_objects, start_label=1)
    label_map[artefacts_filtered == 1] = 0

    assert np.all(filtered_objects.reshape(
        n_objects, -1).sum(axis=1) > area_threshold)
    assert np.all(filtered_objects.sum(axis=0) ^ artefacts_filtered)
    assert np.all(filtered_objects.sum(axis=0) <= 1)
    oh = utils.labelmap2onehot(label_map)
    assert np.all(oh.sum(axis=0) == 1)

    return label_map


def segment_with_fastsam_main(image, n_sp=None, postprocess=True):
    model = FastSAM("FastSAM-x.pt")
    h, w = image.shape[:2]
    if n_sp is None:
        n_sp = int(np.sqrt(h * w))
    area_threshold = int((h * w) / n_sp) / 4
    selem = morph.footprint_rectangle((5, 5))
    tic = time.time()

    results = model(
        image, device="cuda", retina_masks=True, imgsz=1024, conf=0.1, iou=0.7, verbose=VERBOSE
    )
    toc = time.time()

    if VERBOSE:
        print(f"Time FastSAM= {toc - tic:.2f}")
    if results[0].masks is not None:
        mask = results[0].masks.data.cpu()
    else:
        mask = torch.ones((1, h, w))
    if postprocess:
        mask = sort_segmentation_masks(mask)
        label_map = process_segmentation(
            image, [mask], area_threshold, selem, is_not_sam=True
        )
    else:
        label_map = np.array(np.argmax(mask, axis=0))
        label_map += 1

    return label_map


def sort_segmentation_masks(masks: torch.Tensor) -> torch.Tensor:

    pixel_counts = (masks > 0).sum(dim=(-2, -1))
    sorted_indices = torch.argsort(pixel_counts, descending=True)
    sorted_masks = masks[sorted_indices]

    return sorted_masks


def segment_with_sam_main(
    image,
    points_mode,
    points_per_side=32,
    sam_model=None,
    annotations=None,
    image_fname=None,
    n_sp=None,
    postprocess=True,
):

    if sam_model is None:
        sam_model = sam_model_registry["default"](
            checkpoint=sam_model_path).to("cuda")

    # Number of superpixels should be proportional to sqrt(hw)
    # Area threshold should be inversely proportional to expected area of superpixels hw / n_sp
    # Input points for SAM should be sqrt(n_sp) to have n_sp input points
    # Structuring element of arbitrary size
    h, w = image.shape[1:]
    if n_sp is None:
        n_sp = int(np.sqrt(h * w))
    area_threshold = int((h * w) / n_sp) / 4
    selem = morph.footprint_rectangle((5, 5))

    tic = time.time()
    if points_mode == "grid":
        tic = time.time()
        sam_masks = segment_with_sam(
            sam_model, image, area_threshold, points_per_side, area_threshold
        )
        toc = time.time()
        if VERBOSE:
            print(f"Time to segment w/ SAM: {toc - tic:.2f} s")
        tic = time.time()
        if postprocess:
            label_map = process_segmentation(
                image, sam_masks, area_threshold, selem
            )  # SAVE
        else:
            label_map = np.argmax(utils.sammasks2onehot(sam_masks), axis=0)
            label_map += 1
        toc = time.time()
        if VERBOSE:
            print(f"Time to process: {toc - tic:.2f} s")
        if image_fname is not None:
            os.makedirs(f"{out_dir}/grid_{points_per_side}", exist_ok=True)
            np.save(
                f"{out_dir}/grid_{points_per_side}/{image_fname}.npy", label_map)

        if SAVE_INIT:
            image_object = show_objects_with_background(label_map, image)
            plot_grid(image_object, points_per_side)
            plt.axis("off")
            plt.savefig(
                "fig_sam_init_1.png",
                format="png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.show()

    elif points_mode == "gt_center":
        label_map = []
        for a, annotation in enumerate(annotations):
            # SAM segmentation
            tic = time.time()
            center_points = utils.center_points(
                utils.labelmap2onehot(annotation))
            sam_masks_a = segment_with_sam(
                sam_model,
                image,
                area_threshold,
                points_per_side=None,
                point_grids=[center_points[:, ::-1] / [w, h]],
            )
            toc = time.time()
            if VERBOSE:
                print(f"Time to segment: {toc - tic:.2f} s")
            tic = time.time()
            if len(sam_masks_a) > 0:
                label_map_a = process_segmentation(
                    image, sam_masks_a, area_threshold, selem
                )  # SAVE
            else:
                label_map_a = np.ones((image.shape[0], image.shape[1]))
            toc = time.time()
            if VERBOSE:
                print(f"Time to process: {toc - tic:.2f} s")
                print(center_points.shape)
                print(center_points[:, ::-1])
                plt.figure(1)
                plt.subplot(221)
                plt.imshow(annotation)
                plt.scatter(center_points[:, 1],
                            center_points[:, 0], color="red", s=20)
                plt.subplot(222)
                plt.imshow(label_map_a)
                plt.scatter(center_points[:, 1],
                            center_points[:, 0], color="red", s=20)
                plt.subplot(223)
                plt.imshow(image)
                plt.scatter(center_points[:, 1],
                            center_points[:, 0], color="red", s=20)
                plt.show()

            if SAVE_INIT:
                image_object = show_objects_with_background(
                    label_map_a, image, alpha=0.4
                )
                plt.imshow(image_object)
                plt.plot(
                    center_points[:, 1],
                    center_points[:, 0],
                    "o",
                    color=[0, 0, 1],
                    markersize=4,
                )
                plt.axis("off")
                plt.savefig(
                    "tmp/fig_sam_init_2.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.show()

            if image_fname is not None:
                np.save(
                    f"{out_dir}/gt_center/{image_fname}_{a}.npy", label_map_a)
            label_map.append(label_map_a)

    return label_map
