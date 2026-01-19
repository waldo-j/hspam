import argparse
import os
import glob
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as pth_transforms
from inference_dino import infere_one_image, create_model


DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_attention_map(path_folder, arch_type, patch_size, threshold, path_result):
    os.makedirs(path_result, exist_ok=True)
    model = create_model(arch_type, patch_size).eval()
    transform = pth_transforms.ToTensor()
    for img_path in tqdm(sorted(glob.glob(os.path.join(path_folder, "*.jpg")))):
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            tensor = transform(img)
            attentions = infere_one_image(model, threshold, tensor, patch_size)
            fname = os.path.join(path_result, "attn-" +
                                 os.path.basename(img_path))
            plt.imsave(
                fname=fname,
                arr=sum(
                    attentions[i] * 1 / attentions.shape[0]
                    for i in range(attentions.shape[0])
                ),
                cmap="inferno",
                format="jpg",
            )
