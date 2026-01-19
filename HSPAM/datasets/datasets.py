import os
import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from PIL import Image
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from skimage.transform import resize

from ssn.loss import simple_edge_detection, apply_gaussian_blur
from torch.utils.data import Dataset


def normalize_lab(lab_image):
    lab_image[:, :, 0] = lab_image[:, :, 0] / 100  # L: [0, 100] -> [0, 1]
    lab_image[:, :, 1] = (lab_image[:, :, 1]) / \
        128  # a: [-128, 127] -> [-1, 1]
    lab_image[:, :, 2] = (lab_image[:, :, 2]) / \
        128  # b: [-128, 127] -> [-1, 1]
    return lab_image


def denormalize_lab(normalized_lab_image):
    normalized_lab_image[:, :, 0] = (
        normalized_lab_image[:, :, 0] * 100
    )  # L: [0, 1] -> [0, 100]
    normalized_lab_image[:, :, 1] = (
        normalized_lab_image[:, :, 1] * 128
    )  # a: [-1, 1] -> [-128, 127]
    normalized_lab_image[:, :, 2] = (
        normalized_lab_image[:, :, 2] * 128
    )  # b: [-1, 1] -> [-128, 127]
    return normalized_lab_image


def convert_label(label, N_labels=50, shuffle=0):

    onehot = np.zeros(
        (1, N_labels, label.shape[0], label.shape[1])).astype(np.float32)

    set_labels = np.unique(label)
    if shuffle:
        set_labels = np.random.permutation(set_labels)

    ct = 0
    for t in set_labels.tolist():
        if ct >= N_labels:
            break
        else:
            onehot[:, ct, :, :] = label == t
        ct = ct + 1

    return onehot


class BSD(Dataset):
    def __init__(
        self,
        root,
        split="train",
        color_transforms=None,
        geo_transforms=None,
        normalize_img=False,
        get_all_gt=True,
    ):
        self.gt_dir = os.path.join(root, "groundTruth", split)
        self.img_dir = os.path.join(root, "images", split)

        self.index = sorted(os.listdir(self.gt_dir))

        self.color_transforms = color_transforms
        self.geo_transforms = geo_transforms

        self.split = split
        self.normalize_img = normalize_img
        self.get_all_gt = get_all_gt

    def __getitem__(self, idx):
        idx = self.index[idx][:-4]
        gt = scipy.io.loadmat(os.path.join(self.gt_dir, idx + ".mat"))
        t = np.random.randint(0, len(gt["groundTruth"][0]))
        tempo = [gt["groundTruth"][0][0][0][0][0]]
        if self.get_all_gt:
            for i in range(1, len(gt["groundTruth"][0])):
                tempo.append(gt["groundTruth"][0][i][0][0][0])
            gt = np.array(tempo)
        else:
            if gt["groundTruth"].shape[0] > 1:
                gt = gt["groundTruth"]
            else:
                gt = gt["groundTruth"][0][t][0][0][0]

        if os.path.exists(os.path.join(self.img_dir, idx + ".png")):
            img = plt.imread(os.path.join(self.img_dir, idx + ".png"))
        elif os.path.exists(os.path.join(self.img_dir, idx + ".jpg")):
            img = plt.imread(os.path.join(
                self.img_dir, idx + ".jpg")).astype("uint8")
            img = img.astype(np.float32)
            img /= 255

        # COCO images may have sizes around 50
        if not self.get_all_gt and (gt.shape[0] < 270 or gt.shape[1] < 270):
            if gt.shape[0] < gt.shape[1]:
                gt = resize(
                    gt, (270, 270 * gt.shape[1] / gt.shape[0]), order=0)
                img = resize(
                    img, (270, 270 * gt.shape[1] / gt.shape[0]), order=2)
            else:
                gt = resize(
                    gt, (270 * gt.shape[0] / gt.shape[1], 270), order=0)
                img = resize(
                    img, (270 * gt.shape[0] / gt.shape[1], 270), order=2)

        if len(img.shape) == 2:
            img = np.stack((img, img, img), axis=2)

        if len(img.shape) == 2:
            img = np.stack((img, img, img), axis=2)

        ####### USE SAM #####

        img_clean = np.copy(img)
        if self.color_transforms is not None:
            img = self.color_transforms(img)

        img = rgb2lab(img).astype(np.float32)
        if self.normalize_img:
            img = normalize_lab(img)

        gt = gt.astype(np.int64)

        if self.geo_transforms is not None:
            img, gt, img_clean = self.geo_transforms(
                [img, gt, img_clean]
            )

        gt = gt.astype(np.float32)

        # Contours
        if not self.get_all_gt:
            edges = simple_edge_detection(torch.from_numpy(gt[None, :, :]))
            edges_mask = (apply_gaussian_blur(
                edges, kernel_size=7)[0] > 0).float()
            edges = edges[0]
        else:
            edges = np.zeros((gt.shape[0], gt.shape[1]))
            edges_mask = np.zeros((gt.shape[0], gt.shape[1]))

        if not self.get_all_gt:
            if self.split == "val":
                N_labels = 800
                gt = convert_label(gt, N_labels, shuffle=0)
                gt = torch.from_numpy(gt).reshape(N_labels, -1).float()
            else:
                N_labels = 200
                gt = convert_label(gt, N_labels, shuffle=1)
                gt = torch.from_numpy(gt).reshape(N_labels, -1).float()

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        return img, gt, edges, edges_mask, img_clean

    def __len__(self):
        return len(self.index)
