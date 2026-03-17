cd /mnt/petrelfs/tangyuhang/tyh2/code-base/draw_line_for_droid
source ~/miniconda3/bin/activate lerobot
python merge.py \
    --repo-id droid/total \
    --src-path /mnt/petrelfs/tangyuhang/lerobot_data \
    --output-path /mnt/petrelfs/tangyuhang/lerobot_data/droid_total \
    --fps 15 \
    --robot-type franka 