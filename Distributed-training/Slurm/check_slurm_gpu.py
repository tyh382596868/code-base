import os
import socket
import torch  # 必须确保环境里有 pytorch

# --- 1. 获取基础身份信息 ---
proc_id = os.environ.get("SLURM_PROCID", "0")   # 我是第几号任务
num_tasks = os.environ.get("SLURM_NTASKS", "1") # 总共几个任务
node_name = socket.gethostname()                # 我在哪台机器上(节点名)
pid = os.getpid()                               # 我的进程号
# --- 2. 获取 GPU 信息 ---
gpu_available = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if gpu_available else 0

# --- 3. 打印结果 ---
# 为了防止多进程打印时文字乱序，把信息拼成一个长字符串一次性打印
info = f"""
========================================
👋 我是任务 ID: {proc_id} (共 {num_tasks} 个)
📍 所在节点: {node_name} | 进程 PID: {pid}
👀 能看到的 GPU 数量: {gpu_count} 个
"""

if gpu_available:
    for i in range(gpu_count):
        # 获取显卡名称
        p = torch.cuda.get_device_properties(i)
        info += f"   -> GPU {i}: {p.name} (显存: {p.total_memory / 1024**3:.1f} GB)\n"
else:
    info += "   -> ❌ 并没有发现 GPU (CUDA 不可用)\n"

info += "========================================"
print(info)