import copy
import os
import random
from collections import defaultdict, deque
from contextlib import contextmanager

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from far.classes.evaluator import ModelEvaluator
from far.classes.preprocess import Preprocessor
from far.classes.rollout_utils import stack_trajs
from far.diffusion.data4robotics import transforms
from far.diffusion.data4robotics.models.diffusion_unet import \
    DiffusionUnetAgent
from far.critic import Critic, Critic_2head
from far.trainers.bc import BehaviorCloning
from far.trainers.utils import optim_builder, soft_update
import itertools

def get_infinite_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def imgs_to_device(imgs, device):
    """Move a {cam: tensor} dict to device (None for state-only policies)."""
    if imgs is None:
        return None
    return {k: v.to(device) for k, v in imgs.items()}

def flatten_actions(actions):
    """(B, ac_chunk, ac_dim) -> (B, ac_chunk * ac_dim), the action format the critics take."""
    return actions.reshape((actions.shape[0], -1))

def expectile_loss(adv, expectile):
    """IQL asymmetric L2 loss on adv = Q - V."""
    weight = torch.where(adv > 0, expectile, (1 - expectile))
    return torch.mean(weight * (adv ** 2))

def ranking_loss(score_good, score_bad):
    """Bradley-Terry preference loss: -log sigmoid(good - bad)."""
    return -F.logsigmoid(score_good - score_bad).mean()

@contextmanager
def preserved_rng_state():
    """
    Snapshot Python / NumPy / Torch (CPU + all CUDA devices) RNG state and force deterministic cuDNN
    (prevents Conv1d backward from polluting the CUDA context); restore everything on exit.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    orig_deterministic = torch.backends.cudnn.deterministic
    orig_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available() and cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.backends.cudnn.deterministic = orig_deterministic
        torch.backends.cudnn.benchmark = orig_benchmark

class WeigtedActionWrapper:
    IMAGE_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]

    def __init__(self, agent, config, preprocessor: Preprocessor, EXP_WEIGHT=0.0):
        self.config = config
        self.EXP_WEIGHT = EXP_WEIGHT
        self.pred_horizon = config.pred_horizon
        self.weights = np.exp(-self.EXP_WEIGHT * np.arange(self.pred_horizon))
        self.weight_cumsum = np.cumsum(self.weights)
        self.obs_horizon = config.obs_horizon
        self.preprocessor = preprocessor

        if config.dp["num_cams"] == 1:
            self.IMAGE_KEYS = ["agentview_image"]

        self.agent: DiffusionUnetAgent = agent
        self.transform = transforms.get_transform_by_name("preproc")

        assert hasattr(agent, "get_actions")
        self.reset()

    def reset(self):
        # These variables are set the first time get_action is called
        self.image_history = None
        self.act_history = None
        self.num_envs = None
        self.pred_action_history = deque(maxlen=self.pred_horizon)

    def _ensure_history_buffers(self, obs):
        # If buffers are None, initialize them and set num_envs
        if self.image_history is None or self.act_history is None:
            self.num_envs = obs["state"].shape[0]
            self.image_history = defaultdict(
                lambda: [deque(maxlen=self.obs_horizon) for _ in range(self.num_envs)]
            )
            self.act_history = [
                deque(maxlen=self.pred_horizon) for _ in range(self.num_envs)
            ]

    def _apply_weighting(self, acs, weighting, get_full_action):
        if weighting:
            weighted_acts = []
            for env_id in range(self.num_envs):
                ac = self.get_weighted_action(
                    acs[env_id], env_id, get_full_action=get_full_action
                )
                weighted_acts.append(ac)

            return np.stack(weighted_acts, axis=0)
        else:
            return acs if get_full_action else acs[:, 0, :]

    def _get_images_and_states(self, obs):
        """
        Input: num_envs x ...
        Pushes obs into the image history (mutates obs) and returns the preprocessed (images, state).
        """
        return self.get_images_and_states(self.process_obs(obs), training=False)

    def get_weighted_action(self, ac, env_id, get_full_action=False):
        """
        ac: pred_horizon x action_dim
        Return shape:
        - if get_full_action=False: action_dim
        - if get_full_action=True: pred_horizon x action_dim
        """
        # ac: pred_horizon x action_dim
        self.act_history[env_id].append(ac)
        num_actions = len(self.act_history[env_id])

        if not get_full_action:
            # num_actions x action_dim
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self.act_history[env_id]
                    )
                ]
            )

            # more recent predictions get exponentially *less* weight than older predictions
            weights = np.exp(-self.EXP_WEIGHT * np.arange(num_actions))
            weights = weights / weights.sum()

            # compute the weighted average across all predictions for this timestep
            weighted_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        else:
            pred_horizon = ac.shape[0]
            weighted_action = np.zeros_like(ac)
            for i in range(pred_horizon):
                # a(t=i) = (deque[t=0][i]*w[0]== + deque[t=1][1+i]*w[1] + ... + deque[n][n+i]*w[n])/sum(w[:n])
                res_i = np.zeros(ac.shape[1])
                count_i = 0
                for j, pred_actions in zip(
                    range(num_actions - 1, -1, -1), self.act_history[env_id]
                ):
                    if i + j < pred_horizon:
                        res_i += pred_actions[i + j] * self.weights[j]
                        count_i += 1
                weighted_action[i] = res_i / self.weight_cumsum[count_i - 1]
            weighted_action = weighted_action.transpose(
                1, 0
            )  # Shape (action_dim, pred_horizon)

        return weighted_action

    def get_action(self, obs, scale=1.0, weighting=True, get_full_action=False, get_original_action=False, recovery_mode=False):
        """
        obs orignial shape <data_dim> no stacking
        parallel env obs shape: (num_envs, data_dim)
        ~0.12 seconds for fwd pass on 3090
        """
        # If buffers are None, initialize them and set num_envs
        self._ensure_history_buffers(obs)

        images, state = self._get_images_and_states(obs)
        with torch.no_grad():
            # Shape (num_envs, pred_horizon, ...)
            action = self.agent.get_actions(images, state, scale=scale, recovery_mode=recovery_mode)

        acs = action.cpu().numpy().astype(np.float32)

        action = self._apply_weighting(acs, weighting, get_full_action)
        if get_original_action:
            return action, acs
        return action

    def sample_actions(self, obs, scale=1.0, weighting=False, get_full_action=True, num_actions=16, return_pref=True, pref_model=None):
        """
        obs orignial shape <data_dim> no stacking
        parallel env obs shape: (num_envs, data_dim)
        ~0.12 seconds for fwd pass on 3090
        """
        # If buffers are None, initialize them and set num_envs
        self._ensure_history_buffers(obs)

        images, state = self._get_images_and_states(obs)
        with torch.no_grad():
            # Shape (num_envs, pred_horizon, ...)
            action, pref_scores = self.agent.sample_actions(images, state, scale=scale, num_actions=num_actions, return_pref=return_pref, pref_model=pref_model)

        acs = action.cpu().numpy().astype(np.float32)

        action = self._apply_weighting(acs, weighting, get_full_action)
        return action, pref_scores
    
    def get_action_batched(self, obs):
        """
        TODO: fix this for mutliple envs
        This function takes an observation as input,
        and saves the chunks of the predicted actions in a deque.
        The goal is to keep taking that action till the prediction horizon is reached.
        """

        self._ensure_history_buffers(obs)

        images, state = self._get_images_and_states(obs)

        if len(self.pred_action_history) == 0:
            with torch.no_grad():
                actions = (
                    self.agent.get_actions(images, state)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                for i in range(self.pred_horizon):
                    self.pred_action_history.append(actions[:, i, :])

        ac = self.pred_action_history.popleft()
        return ac

    def process_obs(self, obs):
        """
        Push the current images into the per-env history and stack them in place:
        obs[image_key] -> (num_envs, obs_horizon, ...), obs["state"] -> (num_envs, 1, ...).
        """
        # If buffers are None, initialize them and set num_envs
        self._ensure_history_buffers(obs)
            
        assert (
            obs["state"].shape[0] == self.num_envs
        ), "Batch size mismatch in observations"

        # Get stacked images based on images seen by the WeigtedActionWrapper
        if not self.config.state_only:
            stacked_images = defaultdict(
                list
            )  # List of length num_envs, each element is stacked obs_horizon images
            for env_id in range(self.num_envs):
                for key in self.IMAGE_KEYS:
                    # If the length of the deque is less than the obs_horizon, fill with current image
                    if len(self.image_history[key][env_id]) == 0:
                        for _ in range(self.obs_horizon):
                            self.image_history[key][env_id].append(obs[key][env_id])

                    self.image_history[key][env_id].append(obs[key][env_id])

                    # Append Shape (obs_horizon, ...)
                    stacked_images[key].append(
                        np.stack(list(self.image_history[key][env_id]), axis=0)
                    )

            # Stack images across env dim to get shape (num_envs, obs_horizon, ...)
            for key in self.IMAGE_KEYS:
                obs[key] = np.stack(stacked_images[key], axis=0)

        # Unsqueeze state for preprocessing
        obs["state"] = obs["state"][:, None, ...]  # (num_envs, 1, ...)

        return obs
    
    def get_images_and_states(self, processed_obs, training=True):
        # obs = copy.deepcopy(processed_obs)
        obs = processed_obs.copy() 
        obs = self.preprocessor.preprocess_batch(obs, training=training)

        if not self.config.state_only:
            if "robot0_eye_in_hand_image" in obs.keys():
                images = {
                    "cam0": obs["agentview_image"],
                    "cam1": obs["robot0_eye_in_hand_image"],
                }
            else:
                images = {"cam0": obs["agentview_image"]}
        else:
            images = None

        state = obs["state"][:, -1, :].to(torch.float32)  # Take the last state
        return images, state

class DiffusionPolicyAgent:
    def __init__(
        self,
        config,
        diffusion_policy,
        # diffusion_policy: DiffusionBasePolicy,
        noise_std: float,
    ):
        self.config = config
        self.diffusion_policy = diffusion_policy
        self.noise_std = noise_std
        cprint(
            f"Initialized DiffusionPolicyAgent with noise_std: {noise_std}",
            "yellow",
        )

    def get_action(self, obs):
        base_action = self.diffusion_policy.get_action(obs)

        residual_action = np.zeros_like(base_action)

        if self.noise_std > 0:
            residual_action += np.random.normal(
                0, self.noise_std, size=residual_action.shape
            )

        return {
            "base_action": base_action,
            "residual_action": residual_action,
        }

    def reset(self):
        self.diffusion_policy.reset()

class DiffusionBasePolicy:
    def __init__(
        self,
        preprocessor: Preprocessor,
        encoder: nn.Module,
        config,
        device: str,
        state_dim,
        action_dim,
        logger,
        name="DP",
    ):
        super(DiffusionBasePolicy, self).__init__()
        self.config = config
        self.preprocessor = preprocessor
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initialize_model(encoder)
        self.initialize_trainer()
        self.policy = WeigtedActionWrapper(
            agent=self.diffusion_unet,
            config=self.config,
            preprocessor=self.preprocessor,
        )
        self.name = name  # For Logging
        self.logger = logger

    def initialize_model(self, encoder):
        # Initialize gpu_transform
        self.gpu_transform = transforms.get_gpu_transform_by_name("gpu_medium")

        # Instantiate the agent
        diffusion_unet = DiffusionUnetAgent(
            features=encoder,
            shared_mlp=self.config.dp["shared_mlp"],
            odim=self.state_dim,
            n_cams=self.config.dp["num_cams"],
            use_obs=True,
            dropout=0.1,
            train_diffusion_steps=100,
            eval_diffusion_steps=16,
            ac_dim=self.action_dim,
            ac_chunk=self.config.dp["ac_chunk"],
            imgs_per_cam=self.config.dp["img_chunk"],
            share_cam_features=False,
            feat_batch_norm=False,
            noise_net_kwargs={
                # # Original Params
                # "diffusion_step_embed_dim": 256,
                # "down_dims": [256, 512, 1024],
                # Our small config
                # "diffusion_step_embed_dim": 128,
                # "down_dims": [32, 64, 128],
                # DPPO
                "diffusion_step_embed_dim": 16,
                "down_dims": [64, 128, 256],
                "kernel_size": 5,
                "n_groups": 8,
            },
            state_only=self.config.state_only,
        )
        self.diffusion_unet = diffusion_unet

        # Print params in diffusion_unet
        num_params = sum(p.numel() for p in self.diffusion_unet.parameters())
        cprint(f"DiffusionUnet + ResNet18 has {num_params} parameters", "green")

    def _build_critic(self, critic_cls, encoder):
        """Critic / value network sharing the diffusion policy's observation settings."""
        return critic_cls(
            features=encoder,
            shared_mlp=self.config.dp["shared_mlp"],
            odim=self.state_dim,
            n_cams=self.config.dp["num_cams"],
            use_obs=True,
            dropout=0.1,
            ac_dim=self.action_dim,
            ac_chunk=self.config.dp["ac_chunk"],
            imgs_per_cam=self.config.dp["img_chunk"],
            share_cam_features=False,
            feat_batch_norm=False,
            state_only=self.config.state_only,
        )

    def _print_num_params(self, module, label):
        num_params = sum(p.numel() for p in module.parameters())
        cprint(f"{label} has {num_params} parameters", "green")

    def _make_pref_optimizer(self, params):
        return torch.optim.AdamW(
            params,
            lr=self.config.pref["lr"],
            betas=(0.9, 0.999),
            weight_decay=1.0e-6,
            eps=1.0e-8
        )

    def _make_value_optimizer(self, params):
        return torch.optim.AdamW(params, lr=self.config.pref["lr"])
        
    def initialize_critic(self, encoder):
        # Q network (pref_net) + separate V network (v_net) for IQL
        critic_cls = Critic_2head
        self.pref_net = self._build_critic(critic_cls, encoder)
        self.pref_net.to(self.device)
        self._print_num_params(self.pref_net, "Q network + ResNet18")

        pref_params = [param for name, param in self.pref_net.named_parameters() if "value_head" not in name]
        self.pref_optimizer = self._make_pref_optimizer(pref_params)
        
        self.v_net = self._build_critic(critic_cls, copy.deepcopy(encoder))
        self.v_net.to(self.device)
        self.value_optimizer = self._make_value_optimizer(self.v_net.parameters())

    def initialize_trainer(self):
        builder = optim_builder(
            optimizer_type="AdamW",
            optimizer_kwargs={
                "lr": self.config.dp["lr"],
                "betas": [0.95, 0.999],
                "weight_decay": 1.0e-6,
                "eps": 1.0e-8,
            },
        )
        trainer = BehaviorCloning(
            model=self.diffusion_unet,
            device_id=self.device,
            optim_builder=builder,
        )
        self.trainer = trainer

    def _compute_val_bc_loss(self, expert_val_eps):
        """Per-step BC MSE (normalized actions) averaged over validation episodes; 0.0 if expert_val_eps is None."""
        val_total_mse = 0.0
        if expert_val_eps is None:
            return val_total_mse
        val_eps_keys = list(expert_val_eps.keys())
        for key in val_eps_keys:
            # Stack each field and add a dummy BL dimension
            data_traj_i = {k: np.expand_dims(np.stack(v), axis=1) for k, v in expert_val_eps[key].items()}

            (imgs, obs), gt_actions, _ = self.preprocessor.d4r_preprocess_batch(
                data_traj_i, training=False
            )

            # Get the actions from the model
            with torch.no_grad():
                model_actions = self.diffusion_unet.get_actions(imgs, obs)

            # Compute the MSE loss (between normalized actions), normalized by batch size
            val_total_mse += F.mse_loss(
                model_actions, gt_actions, reduction="sum"
            ).item() / len(data_traj_i["state"])
        val_total_mse /= len(val_eps_keys)  # normalize by number of episodes
        print(f"Validation BC Loss: {val_total_mse}")
        return val_total_mse

    def eval_policy(self, eval_envs, expert_val_eps, step=0):
        """
        Evaluate Policy:
        1. Compute BC Loss on expert_val_eps
        2. Evaluate the agent on eval_env and get the success rate
        """
        self.trainer.model.set_eval()
        self.trainer.set_eval()

        val_total_mse = self._compute_val_bc_loss(expert_val_eps)

        # Get the Success Rate
        evaluator = ModelEvaluator(
            agent=self,
            envs=eval_envs,
            default_seed=self.config.seed,
            parent_output_dir=self.config.logdir / f"{self.name}_eval_videos/",
            step=step,
            eval_num_runs=self.config.eval_num_runs,
            visualize=self.config.visualize_eval,
        )
        (
            avg_success_rate,
            avg_total_avg_reward,
            episode_length,
            avg_total_orig_reward,
        ) = evaluator.evaluate_agent()

        self.trainer.set_train()
        self.trainer.model.set_train()

        return val_total_mse, avg_success_rate, avg_total_avg_reward, episode_length

    def reset(self):
        """
        Called when we want to evaluate the policy
        Resets self.policy wrapper
        Copies over EMA weights to model
        """
        self.policy.reset()
        self.trainer.model.set_eval()
        self.trainer.set_eval()
        
    def reset_tta(self):
        """
        Called before the start of retrying with TTA
        Resets self.policy wrapper
        """
        self.policy.reset()
        # self.trainer.model.set_eval() # Don't copy EMA weights for TTA
        self.trainer.model.noise_net.eval()
        self.trainer.set_eval()

    def set_train(self):
        """
        Called after you are done with evaluation
        Restores the model to train mode
        """
        self.trainer.model.set_train()
        self.trainer.set_train()

    def get_action(self, obs, scale=1.0, weighting=True, get_full_action=False, get_original_action=False, recovery_mode=False,
    ):
        """
        obs: B x T x ... (no stacking)
        Policy takes care of all the preprocessing
        """
        # Make a copy of the obs
        obs = copy.deepcopy(obs)
        with torch.no_grad():
            model_actions = self.policy.get_action(
                obs, weighting=weighting, get_full_action=get_full_action, scale=scale, get_original_action=get_original_action, recovery_mode=recovery_mode
            )

        return model_actions
    
    
    def sample_actions(self, obs, scale=1.0, weighting=False, get_full_action=True, num_actions=16, return_pref=True):
        """
        obs: B x T x ... (no stacking)
        Policy takes care of all the preprocessing
        """
        # Make a copy of the obs
        obs = copy.deepcopy(obs)
        # use_pref = False
        with torch.no_grad():
            model_actions, pref_scores = self.policy.sample_actions(
                obs, weighting=weighting, get_full_action=get_full_action, scale=scale, num_actions=num_actions, return_pref=return_pref, pref_model=self.pref_net
            )

        return model_actions, pref_scores

    def get_action_direct(self, obs):
        """
        obs: BS x BL x ... x stack_dim
        Directly call the DDIM sampler to get actions
        Returns of shape BS x BL x pred_horizon x action_dim
        """
        # Make a copy of the obs
        obs = copy.deepcopy(obs)
        assert (
            obs["state"].ndim == 4
        ), "Expected obs['state'] to have 4 dimensions BS x BL x ob_dim x stack_dim"

        # Get the raw actions from the model
        with torch.no_grad():
            BS, BL = obs["state"].shape[0:2]
            (imgs, state), _, _ = self.preprocessor.d4r_preprocess_batch(
                obs, training=False
            )
            direct_action = self.diffusion_unet.get_actions(imgs, state)
            direct_action = einops.rearrange(
                direct_action, "(bs bl) ... -> bs bl ...", bs=BS, bl=BL
            )

        return direct_action

    def get_action_batched(self, obs):
        """
        obs: B x T x ob_chunck x ...
        Policy takes care of all the preprocessing
        """
        # Make a copy of the obs
        obs = copy.deepcopy(obs)

        # Get the actions from the model
        with torch.no_grad():
            model_actions = self.policy.get_action_batched(obs)

        return model_actions

    def _after_bc_step(self, step, log_step, loss, log_prefix):
        """EMA update, periodic loss logging and LR schedule step shared by the BC training loops."""
        trainer = self.trainer
        trainer.model.ema.step(trainer.model.noise_net.parameters())

        if step % self.config.dp["log_freq"] == 0:
            print(f"DP Step: {step}\tLog Step: {log_step}\tLoss: {loss.item()}")
            if self.logger is not None:
                self.logger.scalar(
                    name=f"{log_prefix}/train/loss", value=loss.item()
                )
                self.logger.write(step=log_step, fps=True)

        if step % self.config.dp["schedule_freq"] == 0:
            trainer.step_schedule()

    def _log_eval_metrics(self, log_prefix, log_step, eval_policy_loss, sr, reward, episode_len):
        if self.logger is None:
            return
        self.logger.scalar(name=f"{log_prefix}/eval/l2_loss", value=eval_policy_loss)
        self.logger.scalar(name=f"{log_prefix}/eval/dreamer_success_rate", value=sr)
        self.logger.scalar(name=f"{log_prefix}/eval/dreamer_reward", value=reward)
        self.logger.scalar(name=f"{log_prefix}/eval/dreamer_episode_len", value=episode_len)
        self.logger.write(step=log_step, fps=True, flush=True)

    def train_base_policy(
        self,
        train_dataset,
        expert_val_eps,
        eval_envs,
        init_step=0,
        train_steps=None,
        log_prefix="",
        run_eval=True,
    ):
        """
        Dataset Generator gives data of shape
        obs: B x T x ob_chunck x ...
        acs: B x T x ac_chunk x ...
        expert_val_eps: Validation dataset
        eval_env: Environment for evaluation
        """
        if train_steps is None:
            train_steps = self.config.dp["train_steps"]

        trainer = self.trainer
        trainer.set_train()

        # Move model and EMA to device
        trainer.model.to(trainer.device_id)
        trainer.model.ema.to(trainer.device_id)
        log_step = init_step
        for step in range(train_steps):
            log_step = step + init_step
            batch = next(train_dataset)

            d4r_batch = self.preprocessor.d4r_preprocess_batch(batch)

            # Train for a single step
            trainer.optim.zero_grad()
            loss = trainer.training_step(d4r_batch, step)
            loss.backward()
            trainer.optim.step()

            self._after_bc_step(step, log_step, loss, log_prefix)

            if run_eval and (
                (step % self.config.dp["eval_freq"] == 0 and step > 0)
                or step == train_steps - 1
            ):
                # Evaluate on environment
                (
                    eval_policy_loss,
                    sr,
                    reward,
                    episode_len,
                ) = self.eval_policy(
                    eval_envs=eval_envs, expert_val_eps=expert_val_eps, step=log_step
                )

                self._log_eval_metrics(log_prefix, log_step, eval_policy_loss, sr, reward, episode_len)

                ckpt_file = self.config.logdir / f"{self.name}_base_policy_latest.pt"
                self.ckpt_file = ckpt_file
                trainer.save_checkpoint(ckpt_file, global_step=log_step)

        return log_step + 1

    
    # Advantage Weighted BC
    def train_recovery_policy_awr(
        self,
        train_dataset,
        exp_dataset,
        expert_val_eps,
        eval_envs,
        init_step=0,
        success_dataset=None,
        sampling_probs: list=None,
        train_steps=None,
        log_prefix="",
        run_eval=True,
        epoch=None,
        env_step_train=0,
    ):
        """
        Dataset Generator gives data of shape
        obs: B x T x ob_chunck x ...
        acs: B x T x ac_chunk x ...
        expert_val_eps: Validation dataset
        eval_env: Environment for evaluation
        """
        if train_steps is None:
            train_steps = self.config.dp["train_steps"]
        
        trainer = self.trainer
        trainer.set_train()

        # Move model and EMA to device
        trainer.model.to(trainer.device_id)
        trainer.model.ema.to(trainer.device_id)
        log_step = init_step
        if sampling_probs is not None:
            p_exp, p_rec, p_suc = sampling_probs 
        else:
            p_exp, p_rec, p_suc = 0.7, 0.0, 0.3
        for step in range(train_steps):
            log_step = step + init_step
            rand = random.random()
            if rand < p_exp:
                batch = next(exp_dataset)
                d4r_batch = self.preprocessor.d4r_preprocess_batch(batch)
                loss = trainer.training_step(d4r_batch, step)
            elif rand < p_exp + p_suc:
                batch = next(success_dataset)
                d4r_batch = self.preprocessor.preprocess_batch_awr(batch)            
                loss = trainer.training_step_weighted_bc(d4r_batch, step)
            else:
                batch = next(train_dataset)
                
                d4r_batch = self.preprocessor.preprocess_batch_awr(batch)            
                loss = trainer.training_step_weighted_bc(d4r_batch, step)

            trainer.optim.zero_grad()
            loss.backward()
            trainer.optim.step()

            self._after_bc_step(step, log_step, loss, log_prefix)

            if run_eval and (
                (step % self.config.dp["eval_freq"] == 0 and step > 0)
                or step == train_steps - 1
            ):
                # Evaluate on environment
                (
                    eval_policy_loss,
                    sr,
                    reward,
                    episode_len,
                    env_step_train,
                ) = self.eval_policy_TTA(
                    eval_envs=eval_envs, expert_val_eps=expert_val_eps, step=log_step, env_step_start=env_step_train
                )
                
                self._log_eval_metrics(log_prefix, log_step, eval_policy_loss, sr, reward, episode_len)

                ckpt_file = self.config.logdir / f"{self.name}_recovery_policy_latest.pt"
                self.ckpt_file = ckpt_file
                trainer.save_checkpoint(ckpt_file, global_step=log_step)

        return log_step + 1

    def _prepare_tta_training(self):
        """Put the trainer in train mode on device and freeze the visual encoder (TTA only adapts the rest)."""
        trainer = self.trainer
        trainer.set_train()

        # Move model and EMA to device
        trainer.model.to(trainer.device_id)
        trainer.model.ema.to(trainer.device_id)

        for name, param in trainer.model.visual_feature_net.named_parameters():
            param.requires_grad = False

    def _make_tta_optimizer(self, lr):
        return torch.optim.Adam(
            [p for p in self.trainer.model.parameters() if p.requires_grad],
            lr=lr,
        )

    def _tta_optimizer_step(self, optimizer, total_loss):
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainer.model.parameters(), max_norm=1.0)
        optimizer.step()

    def tta_update_dpo(
        self,
        train_dataset,
        exp_dataset,
        expert_val_eps,
        init_step=0,
        train_steps=10,
        lr=1e-5,
    ):
        """
        Dataset Generator gives data of shape
        obs: B x T x ob_chunck x ...
        acs: B x T x ac_chunk x ...
        expert_val_eps: Validation dataset
        eval_env: Environment for evaluation
        """
        # The global RNG state is restored afterwards so TTA does not shift the rollout's random stream
        with preserved_rng_state():
            print("--- [BEFORE TTA] ---")
            print("NumPy Random:", np.random.rand())

            if train_steps is None:
                train_steps = self.config.dp["train_steps"]
            
            trainer = self.trainer
            self._prepare_tta_training()

            optimizer = self._make_tta_optimizer(lr=lr)
            trainer.model.noise_net.train()
            log_step = init_step

            tta_data = stack_trajs(train_dataset.values(), skip_keys=())
            d4r_batch = self.preprocessor.d4r_preprocess_batch_cl(tta_data, training=False)
            for step in range(train_steps):
                log_step = step + init_step
                optimizer.zero_grad()
                weights = None

                total_loss = trainer.training_step_dpo(d4r_batch, step, weights=weights)
                self._tta_optimizer_step(optimizer, total_loss)
                print(f"DPO Step {step}: Total Loss={total_loss.item():.4f}")

                if step % self.config.dp["schedule_freq"] == 0:
                    trainer.step_schedule()
            
            trainer.model.noise_net.eval()
                
        return log_step + 1
    
    def _load_td_batch(self, batch, device):
        """
        Preprocess a TD batch and move it to device.
        Returns (imgs, obs, ac_flat, next_imgs, next_obs, reward, done); reward and done have shape (B,).
        """
        (imgs, obs), ac, _, (next_imgs, next_obs), reward = self.preprocessor.preprocess_batch_td(batch)
        done = torch.as_tensor(batch["success"], device=device, dtype=torch.float32).squeeze(-1)
        return (
            imgs_to_device(imgs, device),
            obs.to(device),
            flatten_actions(ac.to(device)),
            imgs_to_device(next_imgs, device),
            next_obs.to(device),
            reward.to(device).squeeze(-1),
            done,
        )

    def _load_pref_batch(self, batch, device):
        """Preprocess a preference batch and move it to device. Returns (imgs, obs, ac_good_flat, ac_bad_flat)."""
        (imgs, obs), ac_good, _, ac_bad = self.preprocessor.preprocess_batch_pref(batch)
        return (
            imgs_to_device(imgs, device),
            obs.to(device),
            flatten_actions(ac_good.to(device)),
            flatten_actions(ac_bad.to(device)),
        )

    def _update_value_iql(self, value_net, target_q_values, imgs, obs, expectile):
        """One expectile-regression step of V towards the target Q. Returns (v_pred, v_loss)."""
        v_pred = value_net.get_value(imgs, obs).squeeze(-1)
        v_loss = expectile_loss(target_q_values - v_pred, expectile)

        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()
        return v_pred, v_loss

    @torch.no_grad()
    def _td_target(self, value_net, next_imgs, next_obs, reward, done, gamma):
        """r + gamma * (1 - done) * V(s'), using the freshly updated V network."""
        next_v = value_net.get_value(next_imgs, next_obs).squeeze(-1)
        return reward + gamma * (1 - done) * next_v

    def _save_critic_ckpt(self, output_dir, epoch, state_dicts, tag):
        ckpt_path = os.path.join(output_dir, f"Critic-epoch-{epoch}.pt")
        torch.save(state_dicts, ckpt_path)
        print(f"[{tag}] Saved checkpoint to {ckpt_path}")

    def _remove_prev_critic_ckpt(self, output_dir, epoch):
        prev_ckpt_path = os.path.join(output_dir, f"Critic-epoch-{epoch-1}.pt")
        if os.path.exists(prev_ckpt_path):
            os.remove(prev_ckpt_path)

    def update_critic_iql(
        self,
        train_dataset,
        pref_dataset,
        init_step=0,
        train_steps=None,
        output_dir="scratch_dir/preference_ckpt",
        epoch=None,
        ranking_weight=0.1, 
        expectile=0.7,
        tau=0.005,
        gamma = 0.99
    ):
        """IQL with one two-headed Q network (self.pref_net outputs (q1, q2), clipped double-Q) + V network (self.v_net)."""
        os.makedirs(output_dir, exist_ok=True)
        device = self.config.device
        
        critic = self.pref_net
        critic.train()
        target_critic = copy.deepcopy(critic).to(device)
        target_critic.eval()

        if not hasattr(self, 'v_net'):
            raise ValueError("IQL requires a separate self.v_net")
        value_net = self.v_net
        value_net.train()

        exp_iter = get_infinite_iterator(train_dataset)
        rec_iter = get_infinite_iterator(pref_dataset) if pref_dataset else None
        print(f"\n[IQL Training] Steps: {train_steps}, Expectile: {expectile}")
        for step in range(train_steps):
            log_step = step + init_step
            imgs, obs, ac_flat, next_imgs, next_obs, reward, done = self._load_td_batch(next(exp_iter), device)

            # V update towards min of the two target Q heads (clipped double-Q, against overestimation)
            with torch.no_grad():
                q1_target, q2_target = target_critic(imgs, obs, ac_flat)
                target_q_values = torch.min(q1_target.squeeze(-1), q2_target.squeeze(-1))
            v_pred, v_loss = self._update_value_iql(value_net, target_q_values, imgs, obs, expectile)

            # Q update: TD loss on both heads + ranking loss on both heads
            q_target = self._td_target(value_net, next_imgs, next_obs, reward, done, gamma)
            current_q1, current_q2 = critic(imgs, obs, ac_flat)
            current_q1 = current_q1.squeeze(-1)
            current_q2 = current_q2.squeeze(-1)
            td_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

            rank_loss = torch.tensor(0.0, device=device)
            if rec_iter is not None:
                imgs_rec, obs_rec, ac_good_flat, ac_bad_flat = self._load_pref_batch(next(rec_iter), device)
                q1_good, q2_good = critic(imgs_rec, obs_rec, ac_good_flat)
                q1_bad, q2_bad = critic(imgs_rec, obs_rec, ac_bad_flat)
                rank_loss = (
                    ranking_loss(q1_good.squeeze(-1), q1_bad.squeeze(-1))
                    + ranking_loss(q2_good.squeeze(-1), q2_bad.squeeze(-1))
                )

            total_q_loss = td_loss + ranking_weight * rank_loss

            self.pref_optimizer.zero_grad()
            total_q_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0) 
            self.pref_optimizer.step()
            soft_update(target_critic, critic, tau)
            
            if step % 50 == 0:
                print(f"Step {log_step}: V_Loss={v_loss.item():.5f}, TD_Loss={(td_loss/2).item():.5f}, "
                    f"Rank_Loss={(rank_loss/2).item():.5f}, "
                    f"Q1_mean={current_q1.mean().item():.3f}, V_mean={v_pred.mean().item():.3f}")
    
        self._save_critic_ckpt(output_dir, epoch, {"q_net": critic.state_dict(), "v_net": value_net.state_dict()}, tag="IQL Critic")
        self._remove_prev_critic_ckpt(output_dir, epoch)
        return log_step + 1

    def _value_regression_loss(self, batch, device):
        """MSE between V(s) (pref_net's value head) and batch['value']."""
        (imgs, obs), _, _ = self.preprocessor.d4r_preprocess_batch(batch)
        target = torch.from_numpy(batch['value']).squeeze(-1).to(device).float()
        pred = self.pref_net.get_value(imgs_to_device(imgs, device), obs.to(device)).squeeze(-1)
        return F.mse_loss(pred, target)

    def eval_policy_TTA(self, eval_envs, expert_eps, expert_val_eps, exp_retriever = None, step=0, recovery_buffer=None, success_buffer=None, failure_buffer=None, env_step_start=0):
        """
        Evaluate Policy:
        1. Compute BC Loss on expert_val_eps
        2. Evaluate the agent on eval_env and get the success rate
        """
        self.trainer.model.set_eval()
        self.trainer.set_eval()

        val_total_mse = self._compute_val_bc_loss(expert_val_eps)

        task = self.config.task.split("_")[-1]
        max_retries=self.config.max_retries
        time_limit = self.config.env_time_limits[task]
        # Get the Success Rate
        eval_mode = "rollout" if recovery_buffer is not None else "eval"
        evaluator = ModelEvaluator(
            agent=self,
            envs=eval_envs,
            default_seed=self.config.seed,
            parent_output_dir=self.config.logdir / f"{self.name}_{eval_mode}_videos/",
            step=step,
            eval_num_runs=self.config.eval_num_runs,
            visualize=self.config.visualize_eval,
            exp_retriever=exp_retriever,
            MAX_EPISODE_LENGTH=time_limit,
            max_retries=max_retries,
            env_step_start=env_step_start
        )
        evaluator.expert_eps = expert_eps

        if recovery_buffer is not None:
            # Collect on-policy trajectories
            (
                avg_success_rate,
                avg_total_avg_reward,
                episode_length,
                avg_total_orig_reward,
                env_step_train
            ) = evaluator.rollout_agent_tta(recovery_buffer=recovery_buffer, 
                                            success_buffer=success_buffer,
                                            failure_buffer=failure_buffer)
        else:
            (
                avg_success_rate,
                avg_total_avg_reward,
                episode_length,
                avg_total_orig_reward,
                env_step_train
            ) = evaluator.evaluate_agent_tta()

        self.trainer.set_train()
        self.trainer.model.set_train()

        return val_total_mse, avg_success_rate, avg_total_avg_reward, episode_length, env_step_train
    
    def test_recovery_policy(
        self,
        expert_eps,
        exp_retriever,
        expert_val_eps,
        eval_envs,
        init_step=0,
        train_steps=None,
        log_prefix="",
        run_eval=True,
        estimator=None,
    ):
        """
        Dataset Generator gives data of shape
        obs: B x T x ob_chunck x ...
        acs: B x T x ac_chunk x ...
        expert_val_eps: Validation dataset
        eval_env: Environment for evaluation
        """
        log_step = init_step
        self.estimator = estimator
        # Evaluate on environment (With TTA)
        (
            eval_policy_loss,
            sr,
            reward,
            episode_len,
            env_step_train
        ) = self.eval_policy_TTA(
            eval_envs=eval_envs, expert_eps=expert_eps, expert_val_eps=expert_val_eps, exp_retriever=exp_retriever, step=log_step
        )

        return log_step + 1
    
    def collect_onpolicy_trajs_with_recovery(
        self,
        exp_retriever,
        expert_val_eps,
        eval_envs,
        init_step=0,
        log_prefix="",
        estimator=None,
        recovery_buffer=None,
        success_buffer=None,
        failure_buffer=None,
        env_step_start=0,
    ):
        """
        Dataset Generator gives data of shape
        obs: B x T x ob_chunck x ...
        acs: B x T x ac_chunk x ...
        expert_val_eps: Validation dataset
        eval_env: Environment for evaluation
        """
        log_step = init_step
        self.estimator = estimator
        # Evaluate on environment (With TTA)
        (
            eval_policy_loss,
            sr,
            reward,
            episode_len,
            env_step_train
        ) = self.eval_policy_TTA(
            eval_envs=eval_envs, expert_eps=None, expert_val_eps=expert_val_eps, exp_retriever=exp_retriever, step=log_step, recovery_buffer=recovery_buffer, success_buffer=success_buffer, failure_buffer=failure_buffer, env_step_start=env_step_start
        )

        return env_step_train

    def parameters(self):
        # Return parameters of the Diffusion Model
        return self.trainer.model.parameters()