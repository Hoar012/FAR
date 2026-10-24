# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
from .base import BaseTrainer

class BehaviorCloning(BaseTrainer):
    def training_step(self, batch, global_step):
        (imgs, obs), actions, mask = batch
        if imgs is not None:
            imgs = {k: v.to(self.device_id) for k, v in imgs.items()}
        obs, actions, mask = [ar.to(self.device_id) for ar in (obs, actions, mask)]

        ac_flat = actions.reshape((actions.shape[0], -1))
        mask_flat = mask.reshape((mask.shape[0], -1))
        loss = self.model(imgs, obs, ac_flat, mask_flat)
        # self.log("bc_loss", global_step, loss.item())
        # if self.is_train:
        # self.log("DP/lr", global_step, self.lr)
        return loss

    def training_step_weighted_bc(self, batch, global_step, beta=1.0):
        (imgs, obs), actions, mask, advantage = batch
        if imgs is not None:
            imgs = {k: v.to(self.device_id) for k, v in imgs.items()}
        obs, actions, mask, advantage = [ar.to(self.device_id) for ar in (obs, actions, mask, advantage)]

        ac_flat = actions.reshape((actions.shape[0], -1))
        mask_flat = mask.reshape((mask.shape[0], -1))
        
        adv = advantage
        weights = torch.exp(adv / beta)
        weights = torch.clamp(weights, max=20.0)

        loss = self.model.forward_weighted(imgs, obs, ac_flat, mask_flat, weights=weights)
        # self.log("bc_loss", global_step, loss.item())
        # if self.is_train:
        # self.log("DP/lr", global_step, self.lr)
        return loss
    
    def training_step_dpo(self, batch, global_step, weights=None):
        (imgs, obs), actions, mask, pos_actions = batch
        if imgs is not None:
            imgs = {k: v.to(self.device_id) for k, v in imgs.items()}
            
        obs, actions, mask, pos_actions = [ar.to(self.device_id) for ar in (obs, actions, mask, pos_actions)]
        ac_flat = pos_actions.reshape((pos_actions.shape[0], pos_actions.shape[1], -1))
        
        mask_flat = mask.reshape((mask.shape[0], -1))
        ac_flat_neg = actions.reshape((actions.shape[0], -1))
        loss = self.model.forward_dpo(imgs, obs, ac_flat, mask_flat, ac_flat_neg, weights=weights)
        
        return loss