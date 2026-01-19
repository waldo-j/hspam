import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from skimage.color import lab2rgb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from datasets.datasets import denormalize_lab
from ssn import ssn_iter
from ssn import sparse_ssn_iter


VERBOSE = True


def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1,
                  bias=False, padding_mode="replicate"),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True),
    )


class SSNModel(nn.Module):

    def __init__(self, n_sp, n_iter=10):
        super().__init__()
        self.n_sp = n_sp
        self.n_iter = n_iter
        self.val = 0

    def ssn(self, features, init_label_map, abs_indices):
        if self.training:
            return ssn_iter(
                features,
                self.n_sp,
                self.n_iter,
                init_label_map,
                abs_indices,
                sparse=False,
            )
        else:
            with torch.no_grad():
                return ssn_iter(
                    features,
                    self.n_sp,
                    self.n_iter,
                    init_label_map,
                    abs_indices,
                    sparse=True,
                )


class SSNDefaultCNN(SSNModel):

    def __init__(self, feature_dim, n_sp, n_iter=10):
        super().__init__(n_sp, n_iter)

        self.scale1 = nn.Sequential(conv_bn_relu(5, 64), conv_bn_relu(64, 64))
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1), conv_bn_relu(
                64, 64), conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1), conv_bn_relu(
                64, 64), conv_bn_relu(64, 64)
        )

        self.output_conv = conv_bn_relu(5 + 3 * 64, feature_dim - 5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, init_label_map, abs_indices, inputs_weighted, coords_weighted):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = F.interpolate(
            s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = F.interpolate(
            s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        features_concat = torch.cat([x, s1, s2, s3], 1)
        features = self.output_conv(features_concat)

        self.val += 1
        if VERBOSE and np.mod(self.val, 50000000000000) == 0:
            print(features.shape)
            plt.figure(1)
            tmp = 0
            for i in range(3):
                for j in range(5):
                    plt.subplot(5, 5, tmp + 1)
                    plt.imshow(features[0, tmp, :, :].detach().cpu().numpy())
                    plt.colorbar()
                    tmp += 1
            for j in range(5):
                plt.subplot(5, 5, 15 + j + 1)
                plt.imshow(x[0, j, :, :].detach().cpu().numpy(), cmap="gray")
                plt.colorbar()

        pixel_features = torch.cat(
            [features, inputs_weighted, coords_weighted], 1)

        if VERBOSE and np.mod(self.val, 50000000000) == 0:
            for j in range(5):
                plt.subplot(5, 5, 20 + j + 1)
                plt.imshow(
                    pixel_features[0, 15 + j, :, :].detach().cpu().numpy(), cmap="gray"
                )
                plt.colorbar()
            plt.show()

        return self.ssn(pixel_features, init_label_map, abs_indices)


class SSNCustom(SSNModel):

    def __init__(
        self, n_sp, n_iter=10, n_dims=20, type_backbone="resnet50", use_pca=False
    ):
        super().__init__(n_sp, n_iter)
        self.type_backbone = type_backbone
        self.backbone = self.select_backbone(type_backbone)
        self.n_dims = n_dims
        self.use_pca = use_pca
        self.features = {}
        self._register_hooks()

    def _get_hook(self, layer_name):
        def hook(model, input, output):
            self.features[layer_name] = output.detach()

        return hook

    def select_backbone(self, type_backbone):
        if type_backbone == "resnet50":
            self.conv1x1 = nn.Conv2d(576, 20, kernel_size=1, bias=False)
            return torchvision.models.segmentation.fcn_resnet50(
                weights="DEFAULT",
            ).backbone
        if type_backbone == "resnet101":
            self.conv1x1 = nn.Conv2d(576, 20, kernel_size=1, bias=False)
            return torchvision.models.segmentation.fcn_resnet101(
                weights="DEFAULT"
            ).backbone

        elif type_backbone == "mobilenetv3":
            self.conv1x1 = nn.Conv2d(152, 20, kernel_size=1, bias=False)
            return torchvision.models.mobilenet_v3_large(
                weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
            )
        print("Error type Backbone")
        return torch.Module()

    def __register_hooks_mobilenetv3(self):
        self.backbone.features[1].block[0][2].register_forward_hook(
            self._get_hook("scale1")
        )
        self.backbone.features[2].block[0][2].register_forward_hook(
            self._get_hook("scale2")
        )
        self.backbone.features[3].block[0][2].register_forward_hook(
            self._get_hook("scale3")
        )

    def __register_hooks_resnet(self):
        self.backbone.relu.register_forward_hook(self._get_hook("scale1"))
        self.backbone.layer1[0].relu.register_forward_hook(
            self._get_hook("scale2"))
        self.backbone.layer1[1].relu.register_forward_hook(
            self._get_hook("scale3"))

    def _register_hooks(self):
        if self.type_backbone == "resnet50" or self.type_backbone == "resnet101":
            self.__register_hooks_resnet()
        elif self.type_backbone == "mobilenetv3":
            self.__register_hooks_mobilenetv3()

    def extract_features_resnet50(self, im):
        # Extract features using hooks and resize them
        _ = self.backbone(im)
        s1 = F.interpolate(
            self.features["scale1"],
            size=im.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s2 = F.interpolate(
            self.features["scale2"],
            size=im.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s3 = F.interpolate(
            self.features["scale3"],
            size=im.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat([s1, s2, s3], axis=1)

    def extract_features_resnet101(self, im):
        # Extract features using hooks and resize them
        _ = self.backbone(im)
        s1 = F.interpolate(
            self.features["scale1"],
            size=im.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s2 = F.interpolate(
            self.features["scale2"],
            size=im.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s3 = F.interpolate(
            self.features["scale3"],
            size=im.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat([s1, s2, s3], axis=1)

    def extract_features_mobilenetv3(self, im):
        # Extract features using hooks and resize them
        _ = self.backbone(im)
        s1 = F.interpolate(
            self.features["scale1"],
            size=im.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s2 = F.interpolate(
            self.features["scale2"],
            size=im.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s3 = F.interpolate(
            self.features["scale3"],
            size=im.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat([s1, s2, s3], axis=1)

    def extract_feature(
        self,
        im,
    ):
        if self.type_backbone == "resnet50":
            return self.extract_features_resnet50(im)
        if self.type_backbone == "resnet101":
            return self.extract_features_resnet101(im)
        if self.type_backbone == "mobilenetv3":
            return self.extract_features_mobilenetv3(im)
        print("Error extract features")
        return torch.Tensor()

    def forward(self, x, init_label_map, abs_indices, inputs_weighted, coords_weighted):
        if x.shape[1] == 5:  # image augmented with coords
            im = x[:, :3]
        else:
            im = x

        features = self.extract_feature(im)
        # Apply PCA to reduce dimensions - inference
        # if self.use_pca and not self.training:
        if not self.training and self.use_pca:
            b, c, h, w = features.shape
            # On reshape pour que chaque position spatiale devienne un vecteur de taille (c,)
            features_flattened = (
                features.permute(0, 2, 3, 1).reshape(-1, c).cpu()
            )  # Shape: (b*h*w, c)

            # Appliquer la PCA pour chaque "pixel" (c’est-à-dire chaque vecteur de features par position spatiale)
            pca = PCA(n_components=self.n_dims)
            features_reduced = pca.fit_transform(
                features_flattened
            )  # Shape: (b*h*w, n_dims)

            # Reshape pour retrouver la structure spatiale : (b, h, w, n_dims)
            features_reduced = features_reduced.reshape(b, h, w, self.n_dims)

            # Revenir à la forme tensorielle (b, n_dims, h, w)
            features_reduced = (
                torch.tensor(features_reduced, device=features.device)
                .permute(0, 3, 1, 2)
                .type(torch.float)
            )

        else:
            # Apply conv 1x1 - training
            features_reduced = self.conv1x1(features)

        pixel_features = torch.cat([features_reduced, x], 1)
        return self.ssn(pixel_features, init_label_map, abs_indices)


def plot_feature(tensor_feature, batch=0, channel=0):
    plt.imshow(tensor_feature[batch, channel, :, :].cpu().detach())
    plt.colorbar()
    plt.show()
    return


def create_model(
    type_model, weight, checkpoint, fdim, nspix, n_iter, use_pca
) -> nn.Module:
    if type_model == "ssn":
        print("Load model SSN CNN")
        model = SSNDefaultCNN(feature_dim=fdim, n_sp=nspix, n_iter=n_iter)
    elif type_model == "resnet50":
        print("Load model SSN Resnet 50")
        model = SSNCustom(
            n_sp=nspix,
            n_iter=n_iter,
            n_dims=fdim,
            type_backbone=type_model,
            use_pca=use_pca,
        )
    elif type_model == "resnet101":
        print("Load model SSN Resnet 101")
        model = SSNCustom(
            n_sp=nspix,
            n_iter=n_iter,
            n_dims=fdim,
            type_backbone=type_model,
            use_pca=use_pca,
        )
    elif type_model == "mobilenetv3":
        print("Load model SSN mobilenetv3")
        model = SSNCustom(
            n_sp=nspix,
            n_iter=n_iter,
            n_dims=fdim,
            type_backbone=type_model,
            use_pca=use_pca,
        )
    else:

        def model(data, pre_label_map, abs_indices):
            return sparse_ssn_iter(data, nspix, n_iter, pre_label_map, abs_indices)

        return model

    # Load checkpoint / weights
    if checkpoint is not None:
        try:
            state_dict = torch.load(checkpoint, weights_only=False)
            model.load_state_dict(state_dict["model_state_dict"])
        except:
            raise ValueError("Issue loading checkpoint")

    if weight is not None:
        try:
            model.load_state_dict(torch.load(weight, weights_only=True))
        except:
            raise ValueError("Issue loading weights")

    return model.cuda()
