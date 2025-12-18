cd /mnt/inspurfs/evla2_t/vla_next_next/data_process/code-base/draw_line_for_libero
source /mnt/hwfile/tangyuhang/miniconda3/bin/activate lerobot

python libero_with_traj.py \
    --raw-dir /mnt/inspurfs/evla2_t/vla_next_next/data_process/datasets/libero_goal_no_noops_1.0.0_lerobot \
    --local-dir /mnt/inspurfs/evla2_t/vla_next_next/data_process/datasets_traj \
    --repo-id your_hf_id
