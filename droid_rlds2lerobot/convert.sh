cd /mnt/petrelfs/tangyuhang/tyh2/code-base/droid_rlds2lerobot
source /mnt/hwfile/tangyuhang/miniconda3/bin/activate lerobot
python openx_rlds.py \
  --raw-dir /mnt/petrelfs/tangyuhang/tyh2/DATA/droid/rlds_data/put_the_blue_cube_in_the_red_cup2/tensorflow_datasets/droid/droid/1.0.0 \
  --local-dir /mnt/petrelfs/tangyuhang/tyh2/DATA/droid/standard/droid_lerobot/put_the_blue_cube_in_the_red_cup2 \
  --repo-id your_hf_id \
  --use-videos



