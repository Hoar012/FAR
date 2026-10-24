import os

import imageio
import numpy as np
from scipy import ndimage


def resize_to_given_size(img, final_size):
    # Input shape N x H x W x C (batch of images)
    assert len(img.shape) == 4, "Image should be 4D"
    if img.shape[1] != final_size or img.shape[2] != final_size:
        interp_img = ndimage.zoom(
            img,
            (
                1,
                final_size / img.shape[1],
                final_size / img.shape[2],
                1,
            ),
            order=1,
        )
    else:
        interp_img = img
    return interp_img


def save_frames(frames, filename, frame_successes, suite, task):
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    frame_successes_tiled = np.tile(
        frame_successes[..., None, None], (1, frames.shape[1], frames.shape[2])
    )  # Shape (time, frame_height, frame_width)
    frames[..., 1] = np.clip(
        np.array(frames[..., 1], dtype=np.float32)
        + np.array(frame_successes_tiled, dtype=np.float32) * 20,
        0,
        255,
    )
    filename = os.path.join("demos", f"{suite}__{task}", filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimwrite(
        filename, frames, fps=30, codec="libx264", format="mp4", quality=10
    )
    print("Saved video to", filename, " len", len(frames), " shape", frames.shape)


def save_demo_videos(suite, task, id, frame_successes, agent_frames, robot_frames=None):
    if robot_frames is not None:
        agent_frames = np.concatenate((agent_frames, robot_frames), axis=2)
    save_frames(agent_frames, f"agentview_demo_{id}.mp4", frame_successes, suite, task)
