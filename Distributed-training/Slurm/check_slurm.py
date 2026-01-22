import os
import sys

# 获取 SLURM 环境变量
proc_id = os.environ.get("SLURM_PROCID", "Unknown") # 当前是第几个任务(Rank)
num_tasks = os.environ.get("SLURM_NTASKS", "Unknown") # 总共有多少个任务
pid = os.getpid() # 当前系统的进程号

print(f"👋 大家好! 我是任务 ID: {proc_id} (总共 {num_tasks} 个)。我的系统 PID 是: {pid}")