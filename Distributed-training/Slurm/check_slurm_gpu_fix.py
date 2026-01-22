import os
import socket
import torch

# --- 1. 获取基础身份信息 ---
rank = int(os.environ.get("SLURM_PROCID", "0"))      # 全局 Rank (身份证)
num_tasks = int(os.environ.get("SLURM_NTASKS", "1")) # 总人数
node_name = socket.gethostname()                     # 房间号
pid = os.getpid()                                    # 进程号

# --- 2. 获取 GPU 物理信息 (总共有几把椅子) ---
gpu_available = torch.cuda.is_available()
# 【关键修正】直接问 PyTorch 有几张卡，不依赖不稳定的环境变量
gpus_per_node = torch.cuda.device_count() if gpu_available else 0

# --- 3. 【核心逻辑】计算并绑定座位 ---
current_binding_info = "❌ 未绑定 (CPU模式)"
local_rank = 0

if gpu_available and gpus_per_node > 0:
    # A. 计算：我是这台机器上的第几个人？ (Rank 0->0, Rank 1->1, Rank 4->0 ...)
    local_rank = rank % gpus_per_node
    
    # B. 动作：强制坐下！(这就是“分椅子”的动作)
    torch.cuda.set_device(local_rank)
    
    # C. 验证：现在 PyTorch 认为我当前的主设备是谁？
    current_device_idx = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_device_idx)
    
    # D. 实测：创建一个张量，看它自动落在哪张卡上
    test_tensor = torch.tensor([1]).cuda()
    
    current_binding_info = (
        f"✅ 已绑定逻辑座位: {local_rank}\n"
        f"   -> 验证 current_device(): {current_device_idx}\n"
        f"   -> 验证 Tensor 位置: {test_tensor.device}\n"
        f"   -> 硬件型号: {current_device_name}"
    )

# --- 4. 打印结果 ---
info = f"""
========================================
👋 大家好! 我是全局 Rank: {rank} (进程 PID: {pid})
📍 所在节点: {node_name}

👀 【所见】物理视野:
   在这台机器上，我物理上能看到 {gpus_per_node} 张 GPU。

🪑 【所得】抢椅子结果:
{current_binding_info}
========================================
"""

print(info)