#!/bin/bash

# 激活 lerobot 环境
source /mnt/hwfile/tangyuhang/miniconda3/bin/activate lerobot

# 设置 Python 脚本的路径
SCRIPT_PATH="/mnt/inspurfs/evla2_t/vla_next_next/data_process/code-base/draw_line_for_libero/libero_with_traj.py"

# 数据集根目录
RAW_DIR_BASE="/mnt/inspurfs/evla2_t/vla_next_next/data_process/datasets"

# 你的 Hugging Face ID
REPO_ID="your_hf_id"

# 定义要运行的文件夹列表
DATASET_FOLDERS=(
    "libero_goal_no_noops_1.0.0_lerobot"
    "libero_object_no_noops_1.0.0_lerobot"
    "libero_spatial_no_noops_1.0.0_lerobot"
    "libero_10_no_noops_1.0.0_lerobot"
)

# 遍历每个文件夹并运行 Python 命令
for FOLDER in "${DATASET_FOLDERS[@]}"; do
    RAW_DIR="$RAW_DIR_BASE/$FOLDER"
    LOCAL_DIR="/mnt/inspurfs/evla2_t/vla_next_next/data_process/datasets_traj"

    echo "Running script for $RAW_DIR"

    # 运行 Python 脚本
    python $SCRIPT_PATH \
        --raw-dir $RAW_DIR \
        --local-dir $LOCAL_DIR \
        --repo-id $REPO_ID
done

# 完成后退出环境
conda deactivate


