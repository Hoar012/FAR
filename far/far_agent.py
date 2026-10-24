import collections
import copy
import os

import numpy as np
from termcolor import cprint

from far.classes.preprocess import Preprocessor
from far.classes.resnet_encoder import ResNetEncoder
from far.dreamer import tools
from far.policies.diffusion_policy import (DiffusionBasePolicy,
                                                   DiffusionPolicyAgent)
from far.trainer_utils import (count_n_transitions)
import torch

def get_dynamic_probs(n_exp, n_rec, n_suc):
    """
    Dynamically compute sampling probabilities based on the current size of each dataset.
    """

    # --- 1. Config parameters ---
    # Assume each additional recovery trajectory earns it another 0.5% sampling share.
    prob_per_item = 0.005

    # Cap on the recovery sampling rate (even with 10k+ trajectories, never exceed 30%)
    max_rec_prob = 0.3

    # --- 2. Compute the Recovery probability ---
    if n_rec == 0:
        p_rec = 0.0
    else:
        # Core rule: fewer trajectories -> lower probability; more trajectories -> higher probability, capped.
        # e.g. with just 1 trajectory, p_rec = 0.005 (0.5%) -> safe, avoids overfitting
        # e.g. with 100 trajectories, p_rec = 0.5 (50%) -> hits the cap
        p_rec = min(max_rec_prob, n_rec * prob_per_item)
        p_rec = max(0.01, p_rec)
    # --- 3. Split the remaining probability between Expert and Success ---
    p_remaining = 1.0 - p_rec

    # Here we split the remaining probability 1:1
    p_exp = p_remaining * 0.5
    p_suc = p_remaining * 0.5

    return [p_exp, p_rec, p_suc]


class Recovery_Agent:
    def __init__(
        self,
        config,
        expert_eps,
        state_dim,
        action_dim,
        train_env,
        eval_envs,
        expert_val_eps,
        train_eps,
        init_step,
        logger: tools.Logger = None,
    ):
        self.config = config
        self.expert_eps = expert_eps
        self.train_env = train_env
        self.eval_envs = eval_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expert_val_eps = expert_val_eps
        self.train_eps = train_eps
        self.logger = logger
        self.retriever = None
        
        self.num_expert_transitions = count_n_transitions(self.expert_eps)
        self.expert_dataset = tools.make_dataset(
            self.expert_eps,
            batch_length=1,
            batch_size=self.config.dp["batch_size"]
        )
        
        self.replay_buffer = collections.OrderedDict()
        self.failure_buffer = collections.OrderedDict()
        self._step = init_step
        cprint(f"Initializing Recovery Agent with init_step: {self._step}")
        self._env_step = 0
        
        policy_type = config.policy
        if policy_type == "dp":
            self.base_policy: DiffusionBasePolicy = self.init_dp(load_dp_weights=True)
        
        if self.config.iql_critic["ckpt_path"]:
            self.init_critic(checkpoint_path=self.config.iql_critic["ckpt_path"])
        else:
            self.init_critic()
            cprint("No IQL critic checkpoint provided. Initializing critic from scratch.", "yellow")
            

    def init_dp(self, load_dp_weights=False):
        # Initialize the models
        if self.config.state_only:
            encoder = None
        else:
            encoder = ResNetEncoder(num_cams=self.config.dp["num_cams"])
        preprocessor = Preprocessor(
            config=self.config,
        )
        base_policy = DiffusionBasePolicy(
            preprocessor,
            encoder=encoder,
            config=self.config,
            device=self.config.device,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            name="DP_Distilled",
            logger=self.logger,
        )

        if load_dp_weights:
            if self.config.dp["pretrained_ckpt"] is None:
                raise ValueError("No pretrained checkpoint provided for DP.")

            full_ckpt_path = os.path.join(
                self.config.scratch_dir, self.config.dp["pretrained_ckpt"]
            )
            base_policy.trainer.load_checkpoint(full_ckpt_path)
            
        return base_policy
                    
    def eval_base_policy(self, prefix, round_id, base_policy):
        cprint("\nEvaluating Base Policy", "green")
        # Evaluate the base policy
        eval_policy_loss, sr, reward, episode_len = base_policy.eval_policy(
            eval_envs=self.eval_envs,
            expert_val_eps=self.expert_val_eps,
            step=prefix,
        )
        # Log metrics
        if self.logger is not None:
            num_buffer_transitions = count_n_transitions(self.replay_buffer)
            self.logger.scalar(f"eval/dp_l2_loss", eval_policy_loss)
            self.logger.scalar(f"eval/dp_success_rate", sr)
            self.logger.scalar(f"eval/dp_reward", reward)
            self.logger.scalar(f"eval/dp_episode_len", episode_len)
            self.logger.scalar(f"train/num_buffer_transitions", num_buffer_transitions)
            self.logger.scalar(f"train/env_step", self._env_step)
            self.logger.scalar(f"train/n_round", round_id)
            self.logger.write(step=self._step, fps=True)

    def test_with_tta(self):
        self.estimator = None
        
        cprint(f"\n-------------Starting Evaluation-------------", "green", attrs=["bold"])
        self._step = self.base_policy.test_recovery_policy(
                expert_eps=self.expert_eps,
                exp_retriever=self.retriever,
                expert_val_eps=self.expert_val_eps,
                eval_envs=self.eval_envs,
                init_step=self._step,
                train_steps=self.config.dp["train_steps"],
                log_prefix="recovery_cl_tta",
                run_eval=True,
                estimator=self.estimator,
            )

    # Advantage weighted BC
    @torch.no_grad()
    def filter_data_awr(
            self,
            buffer
        ):
        device = self.config.device
        critic = self.base_policy.pref_net
        v_net = self.base_policy.v_net
    
        traj_ids = list(buffer.keys())
        for traj_id in traj_ids:
            traj = buffer[traj_id]
            train_data = dict()
            for key in traj.keys():
                if key not in train_data:
                    train_data[key] = []
                train_data[key].extend(traj[key])
            for key in train_data.keys():
                train_data[key] = np.stack(train_data[key])[:, np.newaxis, ...]
            
            batch = self.base_policy.preprocessor.d4r_preprocess_batch(train_data, training=False)
            (imgs, obs), actions, mask = batch
        
            if imgs is not None:
                imgs = {k: v.to(device) for k, v in imgs.items()}
            obs, actions, mask = [ar.to(device) for ar in (obs, actions, mask)]      

            arm_action = actions[..., :3]      # [dx, dy, dz, dr...]
            gripper_action = actions[..., -1:]  # [gripper_signal]
            gripper_diff = (gripper_action[:, 1:] - gripper_action[:, :-1]).flatten(1, 2)
            gripper_diff_norm = gripper_diff.norm(p=2, dim=-1)
            arm_ac_flat = arm_action.reshape((arm_action.shape[0], -1))
            action_norm = arm_ac_flat.norm(p=2, dim=-1)
            mask_noop = (gripper_diff_norm < 0.05) & (action_norm < 1)

            # Compute advantage
            q1, q2, value = critic.get_value_with_pref(imgs, obs, actions.reshape((actions.shape[0], -1)))
            q_value = torch.min(q1, q2)

            q_value = q_value.squeeze(-1)
            value = v_net.get_value(imgs, obs).squeeze(-1)
            value = value.squeeze(-1)
            advantage = q_value - value
            mask_keep = ~mask_noop
            mask_keep = np.array(mask_keep.cpu())
            advantage[mask_noop] = -100.0
            traj["advantage"] = advantage.cpu().numpy()
            if mask_keep.sum() == 0:
                del buffer[traj_id]
                continue
            
        return buffer

    # Q-learning dataset
    def preprocess_dataset(self, dataset):
        for traj_id in dataset.keys():
            traj = dataset[traj_id]
            # process image observations to get next_obs
            obs = traj['agentview_image']
            next_obs = obs[1:] + [obs[-1]]
            traj['next_agentview_image'] = next_obs
            if "robot0_eye_in_hand_image" in traj:
                obs_wrist = traj['robot0_eye_in_hand_image']
                next_obs_wrist = obs_wrist[1:] + [obs_wrist[-1]]
                traj['next_robot0_eye_in_hand_image'] = next_obs_wrist
                
            state = traj['state']
            next_state = state[1:] + [state[-1]]
            traj['next_state'] = next_state
            rewards = traj['reward']

            if "advantage" in traj:
                del traj['advantage']
            if "base_action" in traj:
                del traj['base_action']
            if "residual_action" in traj:
                del traj['residual_action']
        return dataset

    def IQL_critic_train(self, output_dir="scratch_dir/critic_ckpt"):
        batch_size = self.config.iql_critic["batch_size"]
        processed_exp = self.preprocess_dataset(self.expert_eps)
        exp_dataset = tools.make_dataset(
                processed_exp,
                batch_length=1,
                batch_size=batch_size,
            )
        
        epochs = self.config.iql_critic["epochs"]        
        training_steps = self.config.iql_critic["warmup_steps"]
        for epoch in range(epochs):
            print(f"[Critic Training] Epoch {epoch}/{epochs}")
            pref_exp = copy.deepcopy(self.expert_eps)
            pref_exp = self.construct_pref_dataset(pref_exp, noise_scale=0.1)
            pref_dataset_exp = tools.make_dataset(
                pref_exp,
                batch_length=1,
                batch_size=batch_size,
            )
            self.base_policy.update_critic_iql(
                train_dataset=exp_dataset,
                pref_dataset=pref_dataset_exp,
                init_step=self._step,
                train_steps=training_steps,
                output_dir=output_dir,
                epoch=epoch,
                ranking_weight=self.config.iql_critic["ranking_weight"],
                expectile=self.config.iql_critic["expectile"],
                tau=self.config.iql_critic["tau"],
                gamma=self.config.iql_critic["gamma"]
            )

    def IQL_policy_train(self):
        self.estimator = None
        recovery_buffer = collections.OrderedDict()
        success_buffer = collections.OrderedDict()
        failure_buffer = collections.OrderedDict()
        
        self.count_list = []
        max_recovery_buffer_size = self.config.max_recovery_buffer_size
        max_success_buffer_size = self.config.max_success_buffer_size
        max_failure_buffer_size = self.config.max_failure_buffer_size

        for epoch in range(self.config.training_epochs):
            # Collect both successful rollouts and unsuccessful rollouts
            cprint(f"\n-------------Starting Continual Policy Improvement Epoch: {epoch}-------------", "green", attrs=["bold"])

            self._env_step = self.base_policy.collect_onpolicy_trajs_with_recovery(
                    exp_retriever=self.retriever,
                    expert_val_eps=self.expert_val_eps,
                    eval_envs=self.eval_envs,
                    init_step=self._step,
                    log_prefix="recovery_dp",
                    estimator=self.estimator,
                    recovery_buffer=success_buffer,
                    success_buffer=success_buffer,
                    failure_buffer=failure_buffer,
                    env_step_start=self._env_step,
                )
            
            if len(recovery_buffer) > max_recovery_buffer_size:
                for traj_id in list(recovery_buffer.keys())[:-max_recovery_buffer_size]:
                    del recovery_buffer[traj_id]

            if len(failure_buffer) > max_failure_buffer_size:
                for traj_id in list(failure_buffer.keys())[:-max_failure_buffer_size]:
                    del failure_buffer[traj_id]

            if len(success_buffer) > max_success_buffer_size:
                for traj_id in list(success_buffer.keys())[:-max_success_buffer_size]:
                    del success_buffer[traj_id]
            print(f"Collected {len(recovery_buffer.keys())} recovery trajectories and {len(success_buffer.keys())} successful trajectories and {len(failure_buffer.keys())} failed trajectories")
            
            for dataset in [self.expert_eps, recovery_buffer, success_buffer, failure_buffer]:
                dataset = self.preprocess_dataset(dataset)
                
            if len(recovery_buffer) > 0:
                critic_dataset = tools.make_mixed_dataset(
                    episodes_list=[self.expert_eps, recovery_buffer, success_buffer, failure_buffer],
                    weights = [0.3, 0.2, 0.2, 0.3],
                    batch_length=1,
                    batch_size=self.config.iql_critic["batch_size"],
                )
            else:
                critic_dataset = tools.make_mixed_dataset(
                    episodes_list=[self.expert_eps, success_buffer, failure_buffer],
                    weights = [0.3, 0.4, 0.3],
                    batch_length=1,
                    batch_size=self.config.iql_critic["batch_size"],
                )

            # Update Critic
            self.base_policy.update_critic_iql(
                train_dataset=critic_dataset,
                pref_dataset=None,
                init_step=self._step,
                train_steps=self.config.iql_critic["train_steps"],
                output_dir=self.config.logdir / f"ft_critic_model",
                epoch=epoch,
                ranking_weight=self.config.iql_critic["ranking_weight"],
                expectile=self.config.iql_critic["expectile"],
                tau=self.config.iql_critic["tau"],
                gamma=self.config.iql_critic["gamma"]
            )

            if epoch < self.config.warmup_epochs:
                continue # Warmup critic for a few epochs before filtering data and training the base policy
            
            self.expert_eps = self.filter_data_awr(self.expert_eps)
            # Use Critic to compute advantages and filter recovery and success buffers
            if len(recovery_buffer) > 0:
                recovery_buffer = self.filter_data_awr(recovery_buffer)
            if len(success_buffer) > 0:
                success_buffer = self.filter_data_awr(success_buffer)

            print(f"After Filtering: {len(recovery_buffer.keys())} recovery trajectories and {len(success_buffer.keys())} successful trajectories")
            
            if len(recovery_buffer) == 0 and len(success_buffer) == 0:
                self._step += self.config.dp["train_steps"]
                continue

            cprint(f"\nTraining Base Policy with Collected Data", "green", attrs=["bold"])
            probs_A = get_dynamic_probs(len(self.expert_eps), len(recovery_buffer), len(success_buffer))
            print(f"Sampling Probabilities - Expert: {probs_A[0]:.3f}, Recovery: {probs_A[1]:.3f}, Success: {probs_A[2]:.3f}")
            batch_size = self.config.dp["batch_size"]
            rec_dataset = tools.make_dataset(
                recovery_buffer,
                batch_length=1,
                batch_size=batch_size,
            )
            succ_dataset = tools.make_dataset(
                success_buffer,
                batch_length=1,
                batch_size=batch_size,
            )
            expert_dataset = tools.make_dataset(
                self.expert_eps,
                batch_length=1,
                batch_size=batch_size,
            )

            if (epoch + 1) % self.config.eval_epochs == 0:
                run_eval = True
            else:
                run_eval = False
            self._step = self.base_policy.train_recovery_policy_awr(
                train_dataset=rec_dataset,
                exp_dataset=expert_dataset,
                expert_val_eps=self.expert_val_eps,
                eval_envs=self.eval_envs,
                init_step=self._step,
                success_dataset=succ_dataset,
                sampling_probs=probs_A,
                ft_steps=self.config.dp["ft_steps"] if self.config.policy == "dp" else 0,
                train_steps=self.config.dp["train_steps"] if self.config.policy == "dp" else 0,
                log_prefix="train_recovery",
                run_eval=run_eval,
                epoch=epoch,
                env_step_train=self._env_step,
            )

    def init_critic(self, checkpoint_path=None):
        encoder = ResNetEncoder(num_cams=self.config.dp["num_cams"])
        if not checkpoint_path and self.config.policy == "dp":
            encoder.load_state_dict(self.base_policy.diffusion_unet.visual_feature_net.state_dict())

        self.base_policy.initialize_critic(encoder=encoder)
    
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path)
            self.base_policy.pref_net.load_state_dict(ckpt["q_net"])
            self.base_policy.v_net.load_state_dict(ckpt["v_net"])
            print(f"Loaded preference head from {checkpoint_path}")

    def construct_pref_dataset(self, train_eps, noise_scale=0.1):
        for traj_id in train_eps.keys():
            traj = train_eps[traj_id]
            actions_traj = traj['action']
            act_neg = []
            T = len(actions_traj)
            for t in range(T):
                a_exp = actions_traj[t]
                a_nonE = a_exp + noise_scale * np.random.randn(*a_exp.shape)
                act_neg.append(a_nonE)
            
            neg_actions = np.array(act_neg)
            train_eps[traj_id]['neg_action'] = neg_actions
        return train_eps