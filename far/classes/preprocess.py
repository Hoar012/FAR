import torch
import torch.nn.functional as F
from einops import rearrange

from far.diffusion.data4robotics import transforms


class Preprocessor:
    DATA_KEYS = ["agentview_image", "state", "robot0_eye_in_hand_image", "action"]

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.gpu_transform = transforms.get_gpu_transform_by_name("gpu_medium")
        self.inference_transform = transforms.get_transform_by_name("preproc")

        if self.config.dp["num_cams"] == 1:
            self.DATA_KEYS = [
                key for key in self.DATA_KEYS if key != "robot0_eye_in_hand_image"
            ]

        if config.state_only:
            # Remove all in DATA_KEYS that contains "image"
            self.DATA_KEYS = [key for key in self.DATA_KEYS if "image" not in key]

    @staticmethod
    def convert_hwc_to_chw_224(img):
        """
        Input can be B x H x W x C or B x T x H x W x C
        Output will be B x C x H x W or B x T x C x H x W
        Resize to 224 if H, W are not 224
        """
        assert len(img.shape) in [4, 5], f"Invalid shape: {img.shape}"
        img = rearrange(img, "... h w c -> ... c h w")

        if img.shape[-1] != 224:
            # Resize each image to 224
            B = img.shape[0]

            # Convert to (B*T, C, H, W) if data has time dimension
            timestep = len(img.shape) == 5
            if timestep:
                img = rearrange(img, "b t c h w -> (b t) c h w")

            img = F.interpolate(
                img, size=(224, 224), mode="bilinear", align_corners=False
            )

            if timestep:
                # Convert back to (B, T, C, H, W)
                img = rearrange(img, "(b t) c h w -> b t c h w", b=B)

        return img

    def preprocess_batch(self, batch, training=True):
        """
        Input shape: BS x some_dim(Optional) x ...
        """
        batch = batch.copy()

        # Normalize images to [0, 1]
        for key, value in batch.items():
            if "image" in key:
                batch[key] = torch.Tensor(value) / 255.0

        batch["cont"] = 1 - torch.Tensor(batch["is_terminal"]).unsqueeze(-1)
        batch = {k: torch.Tensor(v).to(self.device) for k, v in batch.items()}

        # Make images from (H, W, C) to (C, H, W) and resize to 224
        if not self.config.state_only:
            batch["agentview_image"] = self.convert_hwc_to_chw_224(
                batch["agentview_image"]
            )
            if "robot0_eye_in_hand_image" in batch:
                batch["robot0_eye_in_hand_image"] = self.convert_hwc_to_chw_224(
                    batch["robot0_eye_in_hand_image"]
                )

            # Apply GPU Transform to images if training
            if training:
                batch["agentview_image"] = self.gpu_transform(batch["agentview_image"])
                if "robot0_eye_in_hand_image" in batch:
                    batch["robot0_eye_in_hand_image"] = self.gpu_transform(
                        batch["robot0_eye_in_hand_image"]
                    )
            else:
                batch["agentview_image"] = self.inference_transform(
                    batch["agentview_image"]
                )
                if "robot0_eye_in_hand_image" in batch:
                    batch["robot0_eye_in_hand_image"] = self.inference_transform(
                        batch["robot0_eye_in_hand_image"]
                    )

        return batch
    
    def td_preprocess_batch(self, batch, training=True):
        """
        Input shape: BS x some_dim(Optional) x ...
        """
        batch = batch.copy()

        # Normalize images to [0, 1]
        for key, value in batch.items():
            if "image" in key:
                batch[key] = torch.Tensor(value) / 255.0

        batch["cont"] = 1 - torch.Tensor(batch["is_terminal"]).unsqueeze(-1)
        batch = {k: torch.Tensor(v).to(self.device) for k, v in batch.items()}

        # Make images from (H, W, C) to (C, H, W) and resize to 224
        if not self.config.state_only:
            batch["agentview_image"] = self.convert_hwc_to_chw_224(
                batch["agentview_image"]
            )
            batch["next_agentview_image"] = self.convert_hwc_to_chw_224(
                batch["next_agentview_image"]
            )
            if "robot0_eye_in_hand_image" in batch:
                batch["robot0_eye_in_hand_image"] = self.convert_hwc_to_chw_224(
                    batch["robot0_eye_in_hand_image"]
                )
                batch["next_robot0_eye_in_hand_image"] = self.convert_hwc_to_chw_224(
                    batch["next_robot0_eye_in_hand_image"]
                )

            # Apply GPU Transform to images if training
            if training:
                batch["agentview_image"] = self.gpu_transform(batch["agentview_image"])
                batch["next_agentview_image"] = self.gpu_transform(batch["next_agentview_image"])
                if "robot0_eye_in_hand_image" in batch:
                    batch["robot0_eye_in_hand_image"] = self.gpu_transform(
                        batch["robot0_eye_in_hand_image"]
                    )
                    batch["next_robot0_eye_in_hand_image"] = self.gpu_transform(
                        batch["next_robot0_eye_in_hand_image"]
                    )
            else:
                batch["agentview_image"] = self.inference_transform(
                    batch["agentview_image"]
                )
                batch["next_agentview_image"] = self.inference_transform(
                    batch["next_agentview_image"]
                )
                if "robot0_eye_in_hand_image" in batch:
                    batch["robot0_eye_in_hand_image"] = self.inference_transform(
                        batch["robot0_eye_in_hand_image"]
                    )
                    batch["next_robot0_eye_in_hand_image"] = self.inference_transform(
                        batch["next_robot0_eye_in_hand_image"]
                    )

        return batch

    def dreamer_preprocess_batch(self, batch):
        """
        Input: BS x BL x ...
        """
        if "discount" in batch:
            batch["discount"] *= self.config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            batch["discount"] = torch.Tensor(batch["discount"]).unsqueeze(-1)

        return self.preprocess_batch(batch, training=False)

    def encoder_preprocess_batch(self, batch, training=True):
        """
        Input is BS x BL x ... x stack_dim
        Applies preprocessing
        Assumes BL = 2 and stack_dim = 2 for DATA_KEYS
        Returns in format (s, a, s_next, s_priv)
        """
        # Modify from (BS, BL, ..., stack_dim) to (BS*BL, stack_dim, ...) for DATA_KEYS
        # We assume the l = 1 here so do not reshape back to (BS, BL, ...)
        BS = batch["state"].shape[0]
        for key in self.DATA_KEYS:
            batch[key] = rearrange(batch[key], "b l ... s -> (b l) s ...")

        batch = self.preprocess_batch(batch, training=training)

        for key in self.DATA_KEYS:
            batch[key] = rearrange(batch[key], "(b l) ... -> b l ...", b=BS)

        """
        The sampled data is of form
        For BL = 0:
            S_{t-1}, s_t
                     a_t, a_{t+1},...
                 
        For BL = 1:
            s_t, s_{t+1}
                 a_{t+1}, a_{t+2},...
        
        Extracting- (s_t, a_t, s_{t+1}) for training the encoder
        """

        # Assert if BL=2 and obs_stack is 2
        # for key in self.DATA_KEYS:
        #     assert batch[key].shape[1] == 2, f"Shape: {key} - {batch[key].shape}"
        #     assert batch[key].shape[2] == 2, f"Shape: {key} - {batch[key].shape}"

        s = {
            "cam0": batch["robot0_eye_in_hand_image"][:, 1, 0, ...],
            "cam1": batch["agentview_image"][:, 1, 0, ...],
        }

        a = batch["action"][:, 0, 0, ...]
        s_next = {
            "cam0": batch["robot0_eye_in_hand_image"][:, 1, 1, ...],
            "cam1": batch["agentview_image"][:, 1, 1, ...],
        }

        s_priv = batch["privileged_state"][:, 0, ...]
        return s, a.to(torch.float32), s_next, s_priv.to(torch.float32)

    def d4r_preprocess_batch(self, batch, training=True):
        """
        Input is BS x BL x ... x stack_dim
        Applies preprocessing
        Returns with shape BS*BL x stack_dim x ... and format (imgs, obs), actions, mask
        """
        # Modify from (BS, BL, ..., stack_dim) to (BS*BL, stack_dim, ...) for DATA_KEYS
        # We assume the l = 1 here so do not reshape back to (BS, BL, ...)
        for key in self.DATA_KEYS:
            batch[key] = rearrange(batch[key], "b l ... s -> (b l) s ...")

        batch = self.preprocess_batch(batch, training=training)

        # Rename the images
        if not self.config.state_only:
            if "robot0_eye_in_hand_image" in batch:
                imgs = {
                    "cam0": batch["agentview_image"],
                    "cam1": batch["robot0_eye_in_hand_image"],
                }
            else:
                imgs = {
                    "cam0": batch["agentview_image"],
                }
        else:
            imgs = None

        obs = batch["state"][:, -1, :].to(torch.float32)  # Take the last state
        actions = batch["action"].to(torch.float32)
        mask = torch.ones_like(actions).to(torch.float32)  # No mask for now
        return (imgs, obs), actions, mask

    def d4r_preprocess_batch_cl(self, batch, training=True):
        """
        Input is BS x BL x ... x stack_dim
        Applies preprocessing
        Returns with shape BS*BL x stack_dim x ... and format (imgs, obs), actions, mask
        """
        # Modify from (BS, BL, ..., stack_dim) to (BS*BL, stack_dim, ...) for DATA_KEYS
        # We assume the l = 1 here so do not reshape back to (BS, BL, ...)
        
        for key in self.DATA_KEYS:
            batch[key] = rearrange(batch[key], "b l ... s -> (b l) s ...")
        batch['pos_actions'] = rearrange(batch['pos_actions'], "b l m ... s -> (b l) m s ...")

        batch = self.preprocess_batch(batch, training=training)

        # Rename the images
        if not self.config.state_only:
            if "robot0_eye_in_hand_image" in batch:
                imgs = {
                    "cam0": batch["agentview_image"],
                    "cam1": batch["robot0_eye_in_hand_image"],
                }
            else:
                imgs = {
                    "cam0": batch["agentview_image"],
                }
        else:
            imgs = None

        obs = batch["state"][:, -1, :].to(torch.float32)  # Take the last state
        actions = batch["action"].to(torch.float32)
        mask = torch.ones_like(actions).to(torch.float32)  # No mask for now
        pos_actions = batch["pos_actions"].to(torch.float32)
        return (imgs, obs), actions, mask, pos_actions


    def preprocess_batch_pref(self, batch, training=True):
        """
        Input is BS x BL x ... x stack_dim
        Applies preprocessing
        Returns with shape BS*BL x stack_dim x ... and format (imgs, obs), actions, mask
        """
        # Modify from (BS, BL, ..., stack_dim) to (BS*BL, stack_dim, ...) for DATA_KEYS
        # We assume the l = 1 here so do not reshape back to (BS, BL, ...)
        for key in self.DATA_KEYS + ['neg_action']:
            batch[key] = rearrange(batch[key], "b l ... s -> (b l) s ...")

        batch = self.preprocess_batch(batch, training=training)

        # Rename the images
        if not self.config.state_only:
            if "robot0_eye_in_hand_image" in batch:
                imgs = {
                    "cam0": batch["agentview_image"],
                    "cam1": batch["robot0_eye_in_hand_image"],
                }
            else:
                imgs = {
                    "cam0": batch["agentview_image"],
                }
        else:
            imgs = None

        obs = batch["state"][:, -1, :].to(torch.float32)  # Take the last state
        actions = batch["action"].to(torch.float32)
        mask = torch.ones_like(actions).to(torch.float32)  # No mask for now
        neg_actions = batch["neg_action"].to(torch.float32)
        return (imgs, obs), actions, mask, neg_actions
    
    
    def preprocess_batch_awr(self, batch, training=True):
        """
        Input is BS x BL x ... x stack_dim
        Applies preprocessing
        Returns with shape BS*BL x stack_dim x ... and format (imgs, obs), actions, mask
        """
        # Modify from (BS, BL, ..., stack_dim) to (BS*BL, stack_dim, ...) for DATA_KEYS
        # We assume the l = 1 here so do not reshape back to (BS, BL, ...)
        for key in self.DATA_KEYS:
            batch[key] = rearrange(batch[key], "b l ... s -> (b l) s ...")

        batch = self.preprocess_batch(batch, training=training)

        # Rename the images
        if not self.config.state_only:
            if "robot0_eye_in_hand_image" in batch:
                imgs = {
                    "cam0": batch["agentview_image"],
                    "cam1": batch["robot0_eye_in_hand_image"],
                }
            else:
                imgs = {
                    "cam0": batch["agentview_image"],
                }
        else:
            imgs = None

        obs = batch["state"][:, -1, :].to(torch.float32)  # Take the last state
        actions = batch["action"].to(torch.float32)
        mask = torch.ones_like(actions).to(torch.float32)  # No mask for now
        advantage = batch["advantage"].to(torch.float32)
        return (imgs, obs), actions, mask, advantage

    def preprocess_batch_td(self, batch, training=True):
        """
        Input is BS x BL x ... x stack_dim
        Applies preprocessing
        Returns with shape BS*BL x stack_dim x ... and format (imgs, obs), actions, mask
        """
        # Modify from (BS, BL, ..., stack_dim) to (BS*BL, stack_dim, ...) for DATA_KEYS
        # We assume the l = 1 here so do not reshape back to (BS, BL, ...)
        if self.config.dp["num_cams"] == 1:
            next_obs_keys = ['next_agentview_image', 'next_state']
        else:
            next_obs_keys = ['next_agentview_image', 'next_robot0_eye_in_hand_image', 'next_state']
        for key in self.DATA_KEYS + next_obs_keys:
            batch[key] = rearrange(batch[key], "b l ... s -> (b l) s ...")

        batch = self.td_preprocess_batch(batch, training=training)

        # Rename the images
        if not self.config.state_only:
            if "robot0_eye_in_hand_image" in batch:
                imgs = {
                    "cam0": batch["agentview_image"],
                    "cam1": batch["robot0_eye_in_hand_image"],
                }
                next_imgs = {
                    "cam0": batch["next_agentview_image"],
                    "cam1": batch["next_robot0_eye_in_hand_image"],
                }
            else:
                imgs = {
                    "cam0": batch["agentview_image"],
                }
                next_imgs = {
                    "cam0": batch["next_agentview_image"],
                }
        else:
            imgs = None

        obs = batch["state"][:, -1, :].to(torch.float32)  # Take the last state
        actions = batch["action"].to(torch.float32)
        mask = torch.ones_like(actions).to(torch.float32)  # No mask for now
        
    
        next_obs = batch["next_state"][:, -1, :].to(torch.float32)
        reward = batch["reward"].to(torch.float32)
        
        return (imgs, obs), actions, mask, (next_imgs, next_obs), reward
    