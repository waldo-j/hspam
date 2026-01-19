import math
import torch

from ssn.pair_wise_distance import PairwiseDistFunction
from ssn.utils import naive_sparse_bmm


def mean_pooling_by_labels(features, label_map):
    batch_size, num_channels, height, width = features.size()
    num_labels = label_map.max() + 1

    features_flat = features.view(
        batch_size, num_channels, height * width
    )  # (batch_size, num_channels, N)
    label_map_flat = label_map.view(batch_size, height * width).to(
        features.device
    )  # (batch_size, N)

    features_sum = torch.zeros(
        batch_size,
        num_channels,
        num_labels,
        device=features.device,
        dtype=features.dtype,
    )
    features_count = torch.zeros(
        batch_size, num_labels, device=features.device, dtype=torch.float
    )

    # Ça qui prend du temps je pense
    # Pas réussi à le batcher propre (attention les cartes ne sont pas les mêmes)
    for b in range(batch_size):
        indices = label_map_flat[b] + num_labels * torch.arange(
            num_channels, device=features.device
        ).unsqueeze(-1)

        indices_flat = indices.view(-1)
        features_flat_b = features_flat[b].view(-1)

        features_sum_b = features_sum[b].view(num_channels * num_labels)
        features_sum_b.scatter_add_(0, indices_flat, features_flat_b)

        features_count_b = features_count[b]
        features_count_b.scatter_add_(
            0, label_map_flat[b], torch.ones_like(label_map_flat[b], dtype=torch.float)
        )

    features_mean = features_sum / features_count.unsqueeze(1).clamp(min=1)

    return features_mean.view(batch_size, num_channels, num_labels)


def calc_init_centroid(images, init_label_map):
    batchsize, channels, _, _ = images.shape
    with torch.no_grad():
        centroids = mean_pooling_by_labels(images, init_label_map)
    init_label_map = init_label_map.reshape(batchsize, -1).type_as(centroids)
    centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map


# @torch.no_grad()
# def get_abs_indices(init_label_map, centers, k=9, device="cuda"):
#     b, n_pixel = init_label_map.shape
#     #device = init_label_map.device

#     dot_product = torch.mm(centers.to(device), centers.to(device).t()).to(device)
#     _, relative_sp_indices = torch.topk(-dot_product, k=k, largest=False, dim=1)

#     abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
#     abs_spix_indices = relative_sp_indices[init_label_map.long()].permute(0,2,1)

#     abs_spix_indices = abs_spix_indices.reshape(-1).long()
#     abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

#     return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, abs_spix_indices):
    relative_label = affinity_matrix.max(1)[1]
    batch_size, num_pixels = relative_label.shape
    abs_spix_indices = abs_spix_indices.reshape(batch_size, 9, num_pixels)

    batch_indices = torch.arange(batch_size)[:, None].expand(-1, num_pixels)
    pixel_indices = torch.arange(num_pixels)[None, :].expand(batch_size, -1)

    label = abs_spix_indices[batch_indices, relative_label, pixel_indices]

    return label.long()


@torch.no_grad()
def sparse_ssn_iter(pixel_features, num_spixels, n_iter, init_label_map, abs_indices):
    """
    computing assignment iterations with sparse matrix
    detailed process is in Algorithm 1, line 2 - 6
    NOTE: this function does NOT guarantee the backward computation.

    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """
    return ssn_iter(
        pixel_features, num_spixels, n_iter, init_label_map, abs_indices, sparse=True
    )


def ssn_iter(
    pixel_features, num_spixels, n_iter, init_label_map, abs_indices, sparse=False
):
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6

    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """
    batchsize, channels, height, width = pixel_features.shape
    num_spixels_width = int(math.sqrt(num_spixels * width / height))
    num_spixels_height = int(math.sqrt(num_spixels * height / width))

    # init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)
    # abs_indices = get_abs_indices(init_label_map, centers)
    spixel_features, init_label_map = calc_init_centroid(pixel_features, init_label_map)
    abs_spix_indices = abs_indices[1]

    pixel_features_cpy = pixel_features.clone()
    # B x F x H x W
    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1)  # B x HW x F

    if not sparse:
        permuted_pixel_features = permuted_pixel_features.contiguous()

    for _ in range(n_iter):

        dist_matrix = PairwiseDistFunction.apply(
            pixel_features,
            spixel_features,
            abs_spix_indices.float(),
            num_spixels_width,
            num_spixels_height,
        )

        affinity_matrix = (-dist_matrix).softmax(1)
        reshaped_affinity_matrix = affinity_matrix.reshape(-1)

        mask = abs_indices[1] >= 0  # * (abs_indices[1] < num_spixels)
        # que des indices ok normalement
        sparse_abs_affinity = torch.sparse_coo_tensor(
            abs_indices[:, mask], reshaped_affinity_matrix[mask]
        )

        if sparse:
            spixel_features = naive_sparse_bmm(
                sparse_abs_affinity, permuted_pixel_features
            ) / (torch.sparse.sum(sparse_abs_affinity, 2).to_dense()[..., None] + 1e-16)
            abs_affinity = sparse_abs_affinity
        else:
            abs_affinity = sparse_abs_affinity.to_dense().contiguous()
            spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) / (
                abs_affinity.sum(2, keepdim=True) + 1e-16
            )

        spixel_features = spixel_features.permute(0, 2, 1)
        if not sparse:
            spixel_features = spixel_features.contiguous()

    hard_labels = get_hard_abs_labels(affinity_matrix, abs_spix_indices)

    return abs_affinity, hard_labels, spixel_features, pixel_features_cpy


# ############### OLD SSN 2D ####################

# def calc_init_centroid_(images, num_spixels_width, num_spixels_height):
#     """
#     Initialize superpixels label map and centroids.
#     The initial label map is a uniform grid based on the number of superpixels.
#     Image features are average pooled over this uniform grid.

#     Args:
#         images: torch.Tensor
#             A Tensor of shape (B, C, H, W)
#         num_spixels_width: int
#             number of superpixels along width
#         num_spixels_height: int
#             number of superpixels along height

#     Return:
#         centroids: torch.Tensor
#             A Tensor of shape (B, C, N_SP)
#         init_label_map: torch.Tensor
#             A Tensor of shape (B, H * W)
#     """
#     batchsize, channels, height, width = images.shape
#     device = images.device

#     centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))

#     with torch.no_grad():
#         num_spixels = num_spixels_width * num_spixels_height
#         labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
#         init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
#         init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

#     init_label_map = init_label_map.reshape(batchsize, -1)
#     centroids = centroids.reshape(batchsize, channels, -1)

#     return centroids, init_label_map


# @torch.no_grad()
# def get_abs_indices_(init_label_map, num_spixels_width):
#     b, n_pixel = init_label_map.shape
#     device = init_label_map.device
#     # 1D pixel indices, repeated along a 9-neighbors dimension
#     abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
#     # 3x3 relative indices 1D of 9-neighbors superpixels, depending on number of superpixels
#     r = torch.arange(-1, 2.0, device=device)
#     relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
#     # B x 9 x HW map of initial 9-neighbors superpixels
#     abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None])
#     abs_spix_indices = abs_spix_indices.reshape(-1).long()
#     # Batch indices at pixel level, repeated along a 9-neighbors dimension
#     abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()
#     # 3 x (B x 9 x HW)
#     return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


# @torch.no_grad()
# def get_hard_abs_labels_(affinity_matrix, init_label_map, num_spixels_width):
#     relative_label = affinity_matrix.max(1)[1]
#     r = torch.arange(-1, 2.0, device=affinity_matrix.device)
#     relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
#     label = init_label_map + relative_spix_indices[relative_label]
#     return label.long()


# @torch.no_grad()
# def sparse_ssn_iter_(pixel_features, num_spixels, n_iter):
#     """
#     computing assignment iterations with sparse matrix
#     detailed process is in Algorithm 1, line 2 - 6
#     NOTE: this function does NOT guarantee the backward computation.

#     Args:
#         pixel_features: torch.Tensor
#             A Tensor of shape (B, C, H, W)
#         num_spixels: int
#             A number of superpixels
#         n_iter: int
#             A number of iterations
#         return_hard_label: bool
#             return hard assignment or not
#     """
#     return ssn_iter(pixel_features, num_spixels, n_iter, sparse=True)


# def ssn_iter_(pixel_features, num_spixels, n_iter, sparse=False):
#     """
#     computing assignment iterations
#     detailed process is in Algorithm 1, line 2 - 6

#     Args:
#         pixel_features: torch.Tensor
#             A Tensor of shape (B, C, H, W)
#         num_spixels: int
#             A number of superpixels
#         n_iter: int
#             A number of iterations
#         return_hard_label: bool
#             return hard assignment or not
#     """
#     height, width = pixel_features.shape[-2:]
#     num_spixels_width = int(math.sqrt(num_spixels * width / height))
#     num_spixels_height = int(math.sqrt(num_spixels * height / width))

#     spixel_features, init_label_map = \
#         calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
#     abs_indices = get_abs_indices(init_label_map, num_spixels_width)

#     pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
#     permuted_pixel_features = pixel_features.permute(0, 2, 1)
#     if not sparse:
#         permuted_pixel_features = permuted_pixel_features.contiguous()

#     for _ in range(n_iter):
#         dist_matrix = PairwiseDistFunction.apply(
#             pixel_features, spixel_features, init_label_map, num_spixels_width, num_spixels_height)

#         affinity_matrix = (-dist_matrix).softmax(1)
#         reshaped_affinity_matrix = affinity_matrix.reshape(-1)

#         mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
#         sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

#         if sparse:
#             spixel_features = naive_sparse_bmm(sparse_abs_affinity, permuted_pixel_features) \
#                 / (torch.sparse.sum(sparse_abs_affinity, 2).to_dense()[..., None] + 1e-16)
#             abs_affinity = sparse_abs_affinity
#         else:
#             abs_affinity = sparse_abs_affinity.to_dense().contiguous()
#             spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
#                 / (abs_affinity.sum(2, keepdim=True) + 1e-16)

#         spixel_features = spixel_features.permute(0, 2, 1)
#         if not sparse:
#             spixel_features = spixel_features.contiguous()


#     hard_labels = get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)

#     return abs_affinity, hard_labels, spixel_features

##################
