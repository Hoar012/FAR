# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy

import torch
from termcolor import cprint
from torch import nn

# Residual MLP adapted from https://github.com/irom-princeton/dppo/blob/e7f73dffc131570ef7129d5ed1bc98a05cf030ab/model/common/mlp.py#L86
activation_dict = nn.ModuleDict(
    {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity(),
        "Softplus": nn.Softplus(),
    }
)


class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_dict[activation_type]
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def forward(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input


class ResidualMLP(nn.Module):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        if use_layernorm_final:
            self.layers.append(nn.LayerNorm(dim_list[-1]))
        self.layers.append(activation_dict[out_activation_type])

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


class Agent(nn.Module):
    def __init__(
        self,
        feature_net,
        policy,
        shared_mlp,
        odim,
        n_cams,
        use_obs,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        feat_batch_norm=True,
        state_only=False,
    ):
        super().__init__()

        # store visual features (duplicate weights if shared)
        self._share_cam_features = share_cam_features
        if state_only == False:
            self.embed_dim = feature_net.embed_dim  # * n_cams * imgs_per_cam
        else:
            self.embed_dim = 0

        self.visual_feature_net = feature_net

        # store policy network
        self._policy = policy

        # build shared mlp layers
        self._odim = odim if use_obs else 0
        self._use_obs, self._n_cams = bool(use_obs), n_cams
        mlp_in = self._odim + self.embed_dim
        mlp_def = [mlp_in] + shared_mlp

        # layers = [nn.BatchNorm1d(num_features=mlp_in)] if feat_batch_norm else []
        # for i, o in zip(mlp_def[:-1], mlp_def[1:]):
        #     layers.append(nn.Dropout(dropout))
        #     layers.append(nn.Linear(i, o))
        #     layers.append(nn.ReLU())
        # layers.append(nn.Dropout(dropout))
        # self._shared_mlp = nn.Sequential(*layers)

        if len(mlp_def) >= 2:
            # As in https://github.com/irom-princeton/dppo/blob/e7f73dffc131570ef7129d5ed1bc98a05cf030ab/model/diffusion/unet.py#L151
            cprint(f"Using shared mlp with {mlp_def} hidden units", "green")
            self._shared_mlp = ResidualMLP(
                mlp_def,
                activation_type="Mish",
                out_activation_type="Identity",
            )
        else:
            self._shared_mlp = nn.Identity()

        self.obs_enc_dim = mlp_def[-1]
        self.state_only = state_only

    def forward(self, imgs, obs, ac_flat, mask_flat):
        s_t = self._shared_forward(imgs, obs)
        action_dist = self._policy(s_t)
        loss = (
            -torch.mean(action_dist.masked_log_prob(ac_flat, mask_flat))
            if hasattr(action_dist, "masked_log_prob")
            else -(action_dist.log_prob(ac_flat) * mask_flat).sum() / mask_flat.sum()
        )
        return loss

    def get_actions(self, img, obs, zero_std=True):
        policy_in = self._shared_forward(img, obs)
        return self._policy.get_actions(policy_in, zero_std=zero_std)

    def _shared_forward(self, imgs, obs):
        if not self.state_only:
            shared_in = (
                torch.cat((self.embed(imgs), obs), dim=1)
                if self._use_obs
                else self.embed(imgs)
            )
        else:
            shared_in = obs

        return self._shared_mlp(shared_in)

    def embed(self, imgs):
        return self.visual_feature_net(imgs)

    @property
    def odim(self):
        return self._odim

    @property
    def n_cams(self):
        return self._n_cams

    @property
    def ac_chunk(self):
        return self._policy.ac_chunk

    def restore_features(self, restore_path):
        if not restore_path:
            print("No restore path supplied!")
            return
        state_dict = torch.load(restore_path, map_location="cpu")["features"]
        self.visual_feature_net.load_state_dict(state_dict)
        print(f"Restored {restore_path}!")
