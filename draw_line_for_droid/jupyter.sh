srun -p evla2_t \
  --gres=gpu:1 \
  apptainer exec \
    --fakeroot \
    --nv \
    --bind /mnt:/mnt \
    --env MUJOCO_GL=osmesa \
    /mnt/inspurfs/evla2_t/tangyuhang/ENV/simplerenv-vulkan \
    bash -lc '
      source /mnt/petrelfs/tangyuhang/miniconda3/bin/activate lerobot && \
      which python && \ 
      cd /mnt/petrelfs/tangyuhang/tyh2/code-base/draw_line_for_droid && \
      jupyter lab --allow-root --notebook-dir=/mnt/petrelfs/tangyuhang/tyh2/code-base/draw_line_for_droid  --ip=0.0.0.0 --port=10049 --no-browser
    '
