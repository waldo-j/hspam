import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ssn.utils import naive_sparse_bmm, sparse_permute
import numpy as np

VERBOSE = False


def gaussian_kernel(size=5, sigma=1.0):
    coords = torch.tensor([(x - size // 2) ** 2 for x in range(size)])
    grid = coords.unsqueeze(0) + coords.unsqueeze(1)
    kernel = torch.exp(-grid / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # Pour conv2d


def apply_gaussian_blur(edges, kernel_size=3, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma).to(edges.device)
    # print(kernel)
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    edges_blurred = F.conv2d(edges.unsqueeze(
        0), kernel, padding=padding, groups=1)
    return edges_blurred.squeeze(1)


def simple_edge_detection(tensor):
    diff_h = torch.abs(tensor[..., :-1, :-1] - tensor[..., :-1, 1:])
    diff_v = torch.abs(tensor[..., :-1, :-1] - tensor[..., 1:, :-1])
    edges = torch.clamp(diff_h + diff_v, 0, 1)
    return edges


def smooth_edge_detection_old(tensor):
    diff_h = tensor[..., :-1, :-1] - tensor[..., :-1, 1:]
    diff_v = tensor[..., :-1, :-1] - tensor[..., 1:, :-1]
    diff_h = torch.max(diff_h, dim=1).values
    diff_v = torch.max(diff_v, dim=1).values
    diff = torch.stack((diff_v, diff_h), dim=1)
    edges = torch.max(diff, dim=1).values  # to stay between 0 and 1
    return edges


def smooth_edge_detection(tensor):

    diff_h = tensor[..., :-1, :-1] - tensor[..., :-1, 1:]
    diff_v = tensor[..., :-1, :-1] - tensor[..., 1:, :-1]
    edges = torch.sum((diff_h * diff_h + diff_v * diff_v),
                      dim=1) / 4  # sqrt -> NaN

    return edges


def custom_max_pool2d(input_tensor, kernel):

    device = input_tensor.device
    kernel = kernel[0, 0, :, :].to(device)
    kernel_height, kernel_width = kernel.shape
    kernel = kernel.view(1, -1)

    unfolded_input = F.unfold(
        input_tensor, kernel_size=(kernel_height, kernel_width), padding=2
    )
    batch_size, num_patches, num_windows = unfolded_input.shape
    kernel_flat = kernel.expand(batch_size, -1)
    masked_unfolded_input = unfolded_input * kernel_flat.unsqueeze(2)
    max_pooled, _ = masked_unfolded_input.max(dim=1)

    output_height = input_tensor.shape[2]
    output_width = input_tensor.shape[3]
    output_tensor = max_pooled.view(batch_size, 1, output_height, output_width)

    return output_tensor


class FScoreLoss(torch.nn.Module):
    def __init__(self):
        super(FScoreLoss, self).__init__()

        self.kernel = (
            torch.tensor(
                [
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def forward(self, B, B_gt, mode="max", threshold=0.35, n_sp=100):

        B = B.float()
        B_gt = B_gt.float()

        if B.dim() == 3:
            B = B.unsqueeze(1)
        if B_gt.dim() == 3:
            B_gt = B_gt.unsqueeze(1)

        device = B.device

        # Extended borders
        if mode == "max":
            B_f = custom_max_pool2d(B, self.kernel)
            B_gt_f = custom_max_pool2d(B_gt, self.kernel)
        else:
            B_th = torch.clone(B)
            B_th[B_th < threshold] = 0
            B_f = F.conv2d(B_th, self.kernel.to(device), padding=2)
            B_f = (B_f > 0).float()
            B_gt_f = F.conv2d(B_gt, self.kernel.to(device), padding=2)
            B_gt_f = (B_gt_f > 0).float()

        if VERBOSE:
            plt.figure(2)
            plt.subplot(221)
            plt.imshow(B_f[0][0].detach().cpu().numpy())
            plt.subplot(222)
            plt.imshow(B[0][0].detach().cpu().numpy())
            plt.show()

        # BR
        intersection = torch.sum(B_gt * B_f, dim=(2, 3))
        sum_gt = torch.sum(B_gt, dim=(2, 3))
        BR = intersection / (sum_gt + 1e-6)
        # P
        intersection_P = torch.sum(B_gt_f * B, dim=(2, 3))
        sum_pred = torch.sum(B, dim=(2, 3))
        P = intersection_P / (sum_pred + 1e-6)
        # F-score
        Fscore = 2 * P * BR / (P + BR + 1e-6)
        Fscore = P * BR
        Fscore = (P + BR) / 2
        # print(f"{BR} {P} {Fscore}")

        return BR, P, Fscore


def F_contour_loss(Q, C_gt, C_gt_mask, kernel_size=3, mode="max"):

    C_sp_edges = smooth_edge_detection(
        Q.reshape(Q.shape[0], Q.shape[1], C_gt.shape[1] +
                  1, C_gt.shape[2] + 1).float()
    )

    C_sp_edges_mask = C_sp_edges * C_gt_mask

    if VERBOSE:
        plt.figure(1)
        plt.subplot(321)
        plt.imshow(C_sp_edges[0].cpu().detach().numpy())
        plt.title("C_sp_edges[0]")
        plt.subplot(323)
        plt.imshow(C_sp_edges_mask[0].cpu().detach().numpy())
        plt.title("C_sp_edges_mask[0])")
        plt.subplot(325)
        plt.imshow(C_gt_mask[0].cpu().detach().numpy())
        plt.title("C_gt_mask[0])")
        plt.subplot(322)
        plt.imshow(C_gt[0].cpu().detach().numpy())
        plt.title("C_gt[0]")
        plt.subplot(324)
        plt.imshow(C_gt_mask[0].cpu().detach().numpy())
        plt.title("C_gt_mask[0]")
        plt.subplot(326)
        plt.imshow(
            torch.abs(C_gt[0] - C_sp_edges_mask[0]).cpu().detach().numpy())
        plt.title("C_gt[0]-C_sp_edges_mask[0]")
        plt.show()

    loss_fn = FScoreLoss()
    BR, P, F = loss_fn(C_sp_edges_mask, C_gt, mode)
    loss = torch.mean(1 - F)

    return loss


def sparse_reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with the sparse matrix
    NOTE: this function doesn't use it in this project, because may not return correct gradients

    Args:
        assignment: torch.sparse_coo_tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
    spixel_mean = naive_sparse_bmm(assignment, labels) / (
        torch.sparse.sum(assignment, 2).to_dense()[..., None] + 1e-16
    )
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = sparse_permute(assignment, (0, 2, 1))
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = naive_sparse_bmm(
            permuted_assignment, spixel_mean)
    else:
        # index sampling
        reconstructed_labels = torch.stack(
            [sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0
        )
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
    spixel_mean = torch.bmm(assignment, labels) / (
        assignment.sum(2, keepdim=True) + 1e-16
    )
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = assignment.permute(0, 2, 1).contiguous()
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = torch.bmm(permuted_assignment, spixel_mean)
    else:
        # index sampling
        reconstructed_labels = torch.stack(
            [sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0
        )
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def achievable_segmentation_accuracy(superpixel, label):
    """
    Function to calculate Achievable Segmentation Accuracy:
        ASA(S,G) = sum_j max_i |s_j \cap g_i| / sum_i |g_i|

    Args:
        input: superpixel image (H, W),
        output: ground-truth (H, W)
    """

    TP = 0
    unique_id = torch.unique(superpixel)
    for uid in unique_id:
        mask = superpixel == uid
        label_hist = torch.histogram(label[mask].float(), bins=len(unique_id))
        maximum_regionsize = label_hist[0].max()
        TP += maximum_regionsize

    return TP / label.numel()


def reconstruct_loss_with_cross_entropy(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with cross entropy

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment)
    reconstracted_labels = reconstracted_labels / (
        1e-16 + reconstracted_labels.sum(1, keepdim=True)
    )
    mask = labels > 0
    return -(reconstracted_labels[mask] + 1e-16).log().mean()


def reconstruct_loss_with_mse(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with mse

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment)
    return torch.nn.functional.mse_loss(reconstracted_labels, labels)


def contour_loss(Q, C_gt, C_gt_mask, kernel_size=3, sigma=1.0):

    C_sp_edges = torch.clamp(
        torch.sum(
            smooth_edge_detection(
                Q.reshape(
                    Q.shape[0], Q.shape[1], C_gt.shape[1] +
                    1, C_gt.shape[2] + 1
                ).float()
            ),
            1,
        ),
        0,
        1,
    )
    C_sp_edges_mask = C_sp_edges * C_gt_mask
    C_sp_edges_blurred = apply_gaussian_blur(
        C_sp_edges_mask, kernel_size, sigma)

    if VERBOSE:
        plt.figure(1)
        plt.subplot(321)
        plt.imshow(C_sp_edges[0].cpu().detach().numpy())
        plt.title("C_sp_edges[0]")
        plt.subplot(323)
        plt.imshow(C_sp_edges_mask[0].cpu().detach().numpy())
        plt.title("C_sp_edges_mask[0])")
        plt.subplot(325)
        plt.imshow(C_sp_edges_blurred[0].cpu().detach().numpy())
        plt.title("C_sp_edges_blurred[0]")
        plt.subplot(322)
        plt.imshow(C_gt[0].cpu().detach().numpy())
        plt.title("C_gt[0]")
        plt.subplot(324)
        plt.imshow(C_gt_mask[0].cpu().detach().numpy())
        plt.title("C_gt_mask[0]")
        plt.subplot(326)
        plt.imshow(
            torch.abs(C_gt[0] - C_sp_edges_mask[0]).cpu().detach().numpy())
        plt.title("C_gt[0]-C_sp_edges_mask[0]")
        plt.show()

    loss = F.mse_loss(C_sp_edges_mask, C_gt)

    return loss
