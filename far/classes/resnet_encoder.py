import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.getcwd(), "diffusion"))
from data4robotics.models.resnet import RobomimicResNet


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        obs_chunk: int = 2,
        num_cams: int = 2,
        share_cam_features: bool = False,
        device: str = "cpu",
    ):
        super(ResNetEncoder, self).__init__()

        network = RobomimicResNet(
            size=18,
            weights="IMAGENET1K_V1",
            norm_cfg={
                "name": "diffusion_policy",
            },
            img_size=224,
            feature_dim=64,
        ).to(device)

        self._share_cam_features = share_cam_features
        if share_cam_features:
            self.networks = nn.ModuleList([network])
        else:
            self.networks = nn.ModuleList(
                [network] + [copy.deepcopy(network) for _ in range(1, num_cams)]
            )

        self.imgs_per_cam = obs_chunk

    @property
    def embed_dim(self):
        return sum([net.embed_dim * self.imgs_per_cam for net in self.networks])

    def forward(self, imgs):
        def embed_helper(net, im):
            reshaped_im = False
            if len(im.shape) == 5:
                B, T, C, H, W = im.shape
                im = im.reshape((B * T, C, H, W))
                reshaped_im = True

            # Interpolate to (224, 224) if necessary
            # FIXME: @Vib we should put an assert here as this might lead to silent errors
            assert im.shape[-1] == 224
            # im = nn.functional.interpolate(im, size=(224, 224), mode="bilinear")

            # Forward pass
            embeds = net(im)

            # Reshape embeds if necessary
            if reshaped_im:
                embeds = embeds.reshape((B, T * net.embed_dim))

            return embeds

        if self._share_cam_features:
            embeds = [embed_helper(self.networks[0], imgs)]
        else:
            embeds = [
                embed_helper(net, imgs[f"cam{i}"])
                for i, net in enumerate(self.networks)
            ]

        return torch.cat(embeds, dim=1)
