import argparse
import collections
import contextlib
import gc
import os
import pathlib
import sys

sys.path.append(
    os.path.join(os.getcwd(), "far/diffusion")
)  # For diffusion4robotics imports

import numpy as np
import ruamel.yaml as yaml
import torch
from termcolor import cprint

import environments.wrappers as wrappers
import far.dreamer.tools as tools
from environments.concurrent_envs import ConcurrentEnvs
from environments.global_utils import save_demo_videos
from far.classes.preprocess import Preprocessor
from far.classes.resnet_encoder import ResNetEncoder
from far.policies.diffusion_policy import DiffusionBasePolicy
from far.far_agent import Recovery_Agent


# Force EGL rendering in environments
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def train_eval(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()

    # ==================== Logging ====================
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    with open(f"{logdir}/config.yaml", "w") as f:
        yaml.dump(vars(config), f)

    config.logdir = logdir
    log_step = 0
    config.scratch_dir = pathlib.Path(config.scratch_dir).expanduser()
    logger = tools.Logger(config) if config.use_wandb else None

    print("---------------------")
    cprint(f"Task: {config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {config.logdir}", "cyan", attrs=["bold"])

    if config.visualize_eval:
        cprint(
            f"WARNING: Saving videos of evaluation episodes, please turn off if not needed. High resolution render is {config.high_res_render}",
            "red",
            attrs=["bold"],
        )
    print("---------------------")

    # ==================== Create Datasets ====================
    # Check if task can be split by __ else raise error
    if "__" not in config.task:
        raise ValueError(f"Task {config.task} must be of form 'env_suite__task'")

    suite, task = config.task.split("__", 1)
    task = task.lower()
    if suite == "robomimic":
        from environments.robomimic.utils import get_train_val_datasets

        expert_eps, expert_val_eps, _, state_dim, action_dim = get_train_val_datasets(
            config
        )
    elif suite == "robocasa":
        from environments.robocasa.utils import get_train_val_datasets

        expert_eps, expert_val_eps, _, state_dim, action_dim = get_train_val_datasets(
            config
        )
    elif suite == "maniskill":
        from environments.maniskill.utils import \
            get_train_val_datasets_maniskill

        expert_eps, expert_val_eps, _, state_dim, action_dim = (
            get_train_val_datasets_maniskill(config)
        )
    else:
        raise ValueError(f"Unknown env suite {suite}")

    # Set correct values of state_dim and action_dim
    config.state_dim = state_dim
    config.action_dim = action_dim
    cprint(f"Enviroment State Dim: {state_dim}, Action Dim: {action_dim}", "cyan")

    if config.viz_expert_buffer:
        cprint(
            "-----------------Inspecting Expert Dataset, Saving Videos--------------",
            "yellow",
            attrs=["bold"],
        )
        for id, key in enumerate(expert_eps.keys()):
            frame_successes = np.array(expert_eps[key]["success"]).ravel()
            agent_frames = np.array(expert_eps[key]["agentview_image"])[..., -1]
            robot_frames = np.array(expert_eps[key]["robot0_eye_in_hand_image"])[
                ..., -1
            ]
            save_demo_videos(
                suite=suite,
                task=task,
                id=id,
                frame_successes=frame_successes,
                agent_frames=agent_frames,
                robot_frames=robot_frames,
            )
        exit()

    # ==================== Create Envs ====================
    if suite in ["robomimic", "robocasa"]:
        envs = ConcurrentEnvs(
            config=config, env_make=make_env, num_envs=config.num_envs
        )
    elif suite == "maniskill":
        if config.use_cpu_env:
            envs = ConcurrentEnvs(
                config=config, env_make=make_env, num_envs=config.num_envs
            )
        else:
            envs = make_env(config)

    acts = envs.action_space
    print(f"Action Space: {acts}. Low: {acts.low}. High: {acts.high}")
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # ============ Diffusion Policy Pretraining ===============
    # If checkpoint is not provided, train the base policy (diffusion or act) from scratch
    if config.dp["pretrained_ckpt"] == "":
        cprint(
            "----------------No base policy path provided, begin training diffusion policy--------------",
            "yellow",
            attrs=["bold"],
        )
        # Initialize DP
        preprocessor = Preprocessor(config=config)
        encoder = None if config.state_only else ResNetEncoder(num_cams=config.dp["num_cams"])
        
        policy_type = config.policy
        if policy_type == "dp":
            base_policy = DiffusionBasePolicy(
                preprocessor=preprocessor,
                encoder=encoder,
                config=config,
                device=config.device,
                state_dim=state_dim,
                action_dim=action_dim,
                logger=logger,
                name="DP_Pretrain",
            )

        # Train it
        expert_dataset_dp = tools.make_dataset(
            expert_eps, batch_length=1, batch_size=config.dp["batch_size"]
        )
        log_step = base_policy.train_base_policy(
            train_dataset=expert_dataset_dp,
            expert_val_eps=expert_val_eps,
            eval_envs=envs,
            log_prefix=f"{policy_type}_pretrain",
            run_eval=True
        )

        # Store the saved pretrained checkpoint path
        config.dp["pretrained_ckpt"] = os.path.relpath(
            base_policy.ckpt_file, config.scratch_dir
        )

        # Cleanup
        del preprocessor
        del encoder
        del base_policy
        torch.cuda.empty_cache()
        gc.collect()
        
    # ============ Failure-Aware Retry ==============
    if config.recovery:

        # Create buffer
        train_eps = collections.OrderedDict()

        # Build trainer
        trainer = Recovery_Agent(
            config=config,
            expert_eps=expert_eps,
            state_dim=state_dim,
            action_dim=action_dim,
            train_env=envs,
            eval_envs=envs,
            expert_val_eps=expert_val_eps,
            train_eps=train_eps,
            init_step=log_step,
            logger=logger,
        )
        if config.warmup_critic:
            cprint(
                "\n-----------------Pretraining Critic--------------",
                "yellow",
                attrs=["bold"],
            )
            trainer.IQL_critic_train(output_dir=str(config.logdir / "critic_ckpt"))
        elif config.policy_improvement:
            cprint(
                "\n-----------------Continual Policy Improvement--------------",
                "yellow",
                attrs=["bold"],
            )
            trainer.IQL_policy_train()
        elif config.eval_recovery:
            cprint(
                "\n-----------------Begin testing Failure-Aware Retry--------------",
                "yellow",
                attrs=["bold"],
            )
            trainer.test_with_tta()
            
        # Cleanup
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    envs.close()
    cprint("--------Finished Everything--------", "yellow", attrs=["bold"])


def close_envs(envs):
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def make_env(config):
    suite, task = config.task.split("__", 1)
    task = task.lower()
    if suite == "robomimic":
        from environments.robomimic.constants import IMAGE_OBS_KEYS
        from environments.robomimic.env_make import make_env_robomimic
        from environments.robomimic.utils import (
            create_shape_meta, get_robomimic_dataset_path_and_env_meta)

        dataset_path, env_meta = get_robomimic_dataset_path_and_env_meta(
            env_id=task,
            shaped=config.shape_rewards,
            image_size=config.image_size,
            done_mode=config.done_mode,
            datadir=config.datadir,
        )
        shape_meta = create_shape_meta(img_size=config.image_size, include_state=True)

        shape_rewards = config.shape_rewards
        env = make_env_robomimic(
            env_meta,
            IMAGE_OBS_KEYS,
            shape_meta,
            add_state=True,
            reward_shaping=shape_rewards,
            config=config,
            offscreen_render=False,
        )
        env = wrappers.TimeLimit(env, duration=config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)

    elif suite == "robocasa":
        from environments.robocasa.utils import make_env_robocasa

        env = make_env_robocasa(
            config=config,
            task=task,
            suite=suite,
        )
        env = wrappers.TimeLimit(env, duration=config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)

    elif suite == "maniskill":
        from environments.maniskill.utils import make_maniskill_env

        env = make_maniskill_env(config, suite=suite, task=task)
        env = wrappers.UUID(env)

    else:
        raise ValueError(f"Unknown env suite {suite}")

    return env


# Function to update nested dictionary values based on multi-level keys
def update_nested_obj(obj, key_str, value):
    keys = key_str.split(".")

    # Traverse down the nested object attributes or dictionary
    for key in keys[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]  # Handle dictionary
        else:
            obj = getattr(obj, key)  # Handle object attributes

    # Update the final key/attribute
    final_key = keys[-1]
    if isinstance(obj, dict):
        # Throw error if key is not present in dictionary
        if final_key not in obj:
            raise KeyError(f"Key '{keys}' not found in dictionary during --set")
        obj[final_key] = value  # Set value for dictionary
    else:
        if not hasattr(obj, final_key):
            raise AttributeError(f"Attribute '{keys}' not found in object during --set")
        setattr(obj, final_key, value)  # Set value for object attribute

    print(f"Set {key_str} to {value}")


# Function to convert string values from --set to appropriate type
def convert_type(value):
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Return as string if it's neither int nor float


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="far/configs/far.yaml")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    args, remaining = parser.parse_known_args()

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(pathlib.Path(args.config_path).read_text())

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    # Use nargs='+' to collect multi-level keys and values
    parser.add_argument(
        "--set",
        nargs="+",
        action="append",
        help="Set a configuration key, e.g. --set residual_training.actor_loss_anneal True",
    )

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_known_args(remaining)[0]

    final_config.dp["ac_chunk"] = final_config.pred_horizon

    # Update args passed through --set
    if final_config.set:
        for set_arg in final_config.set:
            key_str = set_arg[0]  # e.g., "residual_training.actor_loss_anneal"
            value_str = set_arg[1]  # e.g., "True"
            value = convert_type(value_str)
            update_nested_obj(final_config, key_str, value)

    # Set Wandb Stuff
    exp_name = f"{str(final_config.task).lower()}/{final_config.wandb_exp_name}_demos{final_config.num_exp_trajs}"
    final_config.wandb_exp_name = f"{exp_name}"

    # Set time limit
    suite, task = final_config.task.split("__", 1)
    task = task.lower()
    max_retries = final_config.max_retries
    if suite not in ("robomimic", "robocasa", "maniskill"):
        raise ValueError(f"Unknown env suite {suite}")
    final_config.time_limit = final_config.env_time_limits[task] * max_retries

    # Set log dir and datadir
    final_config.logdir = (
        f"{final_config.scratch_dir}/logs/{exp_name}/seed{final_config.seed}"
    )
    final_config.datadir = os.path.join("datasets", f"{suite}_datasets")

    if final_config.generate_highres_eval:
        final_config.high_res_render = True

    train_eval(final_config)
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        gc.collect()  # Force garbage collection to run
