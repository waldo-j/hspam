import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms as pth_transforms
import DINO.dino_model as dino_model

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_model(arch_type, patch_size):

    url = None
    model = None
    if arch_type == "vit_tiny":
        pass
    elif arch_type == "vit_small":
        if patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            model = dino_model.vit_small(patch_size)
        elif patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            model = dino_model.vit_small(patch_size)
    elif arch_type == "vit_base":
        if patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            model = dino_model.vit_base(patch_size)
        elif patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            model = dino_model.vit_base(patch_size)
    else:
        print("Error arch type")

    if model is None or url is None:
        print("Error wrong config")
        return nn.Module()
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/dino/" + url
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE)


def infere_one_image(model, threshold, tensor, patch_size):
    transform = pth_transforms.Compose(
        [
            pth_transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    tensor_normalized = transform(tensor[0, :3])  # Only get RGB images

    # make the image divisible by the patch size
    w, h = (
        tensor_normalized.shape[1] - tensor_normalized.shape[1] % patch_size,
        tensor_normalized.shape[2] - tensor_normalized.shape[2] % patch_size,
    )
    tensor_normalized = tensor_normalized[:, :w, :h].unsqueeze(0)

    w_featmap = tensor_normalized.shape[-2] // patch_size
    h_featmap = tensor_normalized.shape[-1] // patch_size

    attentions = model.get_last_selfattention(tensor_normalized.to(DEVICE))

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = (
        nn.functional.interpolate(
            th_attn.unsqueeze(0),
            scale_factor=patch_size,
            mode="nearest",
        )[0]
        .cpu()
        .detach()
    )

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0),
            scale_factor=patch_size,
            mode="nearest",
        )[0]
        .cpu()
        .detach()
    )
    return attentions
