import robosuite as suite
from termcolor import cprint

from environments.robomimic.robosuite_image_wrapper import \
    RobosuiteImageWrapper


def make_env_robomimic(
    env_meta,
    obs_keys,
    shape_meta,
    config,
    add_state=True,
    reward_shaping=False,
    offscreen_render=False,
    has_renderer=True,
):
    """
    Create and configure an environment based on the provided metadata.

    Args:
        env_meta (dict): Metadata containing environment configuration.
        obs_keys (list): List of observation keys.
        shape_meta (dict): Metadata containing shape configuration.
        add_state (bool, optional): Flag indicating whether to add state information.
            Note: only used for image-based experiments
        offscreen_render (bool, optional): Flag indicating whether to enable offscreen rendering. Defaults to False.

    Returns:
        env: Configured environment object.

    """
    env_kwargs = env_meta["env_kwargs"]
    env_name = env_meta["env_name"]

    # Robosuite's hard reset causes excessive memory consumption and significantly
    # increases training time. Assumed True if not set; therefore, disabled here.
    # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
    env_kwargs["hard_reset"] = False
    env_kwargs["env_name"] = env_name
    env_kwargs["has_offscreen_renderer"] = (
        env_kwargs["has_offscreen_renderer"] or offscreen_render
    )
    if has_renderer:
        env_kwargs["has_renderer"] = True
        env_kwargs["render_camera"] = "agentview"
    env_kwargs["reward_shaping"] = reward_shaping

    if config.high_res_render:
        env_kwargs["camera_heights"] = config.highres_img_size
        env_kwargs["camera_widths"] = config.highres_img_size

    env = suite.make(**env_kwargs)
    env = RobosuiteImageWrapper(
        env, keys=obs_keys, shape_meta=shape_meta, add_state=add_state, config=config
    )
    cprint(
        f"Initialized robomimic env with action repeat: {config.action_repeat}, time limit: {config.time_limit}",
        "yellow",
    )
    return env
