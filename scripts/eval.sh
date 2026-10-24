# Only for ManiSkill
# export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json # Please change this to your vulkan path

SUITE="robomimic" # [robomimic | maniskill | robocasa]
TASK="lift" # Any of the tasks in the paper for the respective suite
NUM_EXP_TRAJS=5

SEED=0
python3 main.py --wandb_exp_name "eval_recovery" \
    --configs ${SUITE} cfg_far cfg_eval_recovery\
    --task "${SUITE}__${TASK}" \
    --num_exp_trajs ${NUM_EXP_TRAJS} \
    --seed ${SEED} \
    --set dp.pretrained_ckpt "" \
    --set iql_critic.ckpt_path ""