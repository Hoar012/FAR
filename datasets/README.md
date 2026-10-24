# Expert Demonstrations
We provide details on using the collected demonstrations to train SAILOR.

## Downloading datasets
The collected demonstrations is available for download at this [link](https://huggingface.co/datasets/arnavkj1995/SAILOR/tree/main). Use the following command to download the datasets:
```bash
cd datasets && python download_datasets.py && cd ..
```

The dataset should be downloaded in the datasets folder of the code with the final directory structure being
```
SAILOR
├── datasets
│   ├── maniskill_datasets
│   ├── robomimic_datasets
│   ├── robocasa_datasets
└── ...
```

## Preprocessing datasets
Ensure that the conda environments for each suite are installed. If not, please refer to the main README for setup instructions. The commands below convert the raw datasets downloaded by the commands above to RGB observations and proprioceptive states, and store these trajectories in the same folder as the raw data for the task.

For tasks in `RoboMimic`:
```bash
conda activate robomimic_env
TASK=can
python3 -m robomimic.scripts.dataset_states_to_obs \
    --done_mode 1 \
    --dataset datasets/robomimic_datasets/${TASK}/ph/demo_v141.hdf5 \
    --output_name image_64_shaped_done1_v141.hdf5 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 64 \
    --camera_width 64 \
    --shaped
```

For tasks in `ManiSkill`:
```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json # Please change this to your vulkan path
conda activate maniskill_env

TASK=pullcube
python3 environments/maniskill/convert_data.py \
    --traj-path datasets/maniskill_datasets/${TASK}/demo.h5 \
    --use-first-env-state -c pd_ee_delta_pose -o rgb \
    --save-traj --num-envs 10 -b physx_cpu \
    --record_rewards --reward_mode dense 
```

For tasks in `RoboCasa`:
```bash
conda activate robocasa_env

TASK=bread
python3 environments/robocasa/convert_data.py --done_mode 1 \
  --dataset datasets/robocasa_datasets/${TASK}/demo.hdf5 \
  --output_name image_64_shaped_done1_v141.hdf5 \
  --camera_names agentview robot0_eye_in_hand --camera_height 64 --camera_width 64 --shaped
```