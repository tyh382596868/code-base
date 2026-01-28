import subprocess
import sys
import os
import platform
import shutil

# ==========================================
# 用户配置区域 (修改这里以适配不同项目)
# ==========================================

CONFIG = {
    "project_name": "openvla",  # 对应 conda create -n openvla
    "python_version": "3.10",
    "use_uv": False,                # 该脚本使用标准 pip，未使用 uv

    # # === [新增] 全局镜像源配置 ===
    # # 只要配在这里，下面所有的 pip install 都不用再写 -i 参数了
    # "global_env": {
    #     "PIP_INDEX_URL": "https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/",
    #     "PIP_TRUSTED_HOST": "pkg.pjlab.org.cn"
    # },

    # === [关键修改] 全局环境变量配置 ===
    "global_env": {
        # 1. 配置 pip 镜像源
        "PIP_INDEX_URL": "https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/",
        "PIP_TRUSTED_HOST": "pkg.pjlab.org.cn",

        # 2. 【核心解决方案】配置不走代理的白名单
        # 告诉 pip 和其他工具：访问 pjlab.org.cn 时，不要经过代理
        # localhost,127.0.0.1 是常规建议，防止本地通讯也走代理出问题
        "no_proxy": "localhost,127.0.0.1,pkg.pjlab.org.cn",
        "NO_PROXY": "localhost,127.0.0.1,pkg.pjlab.org.cn" 
    },

    # 1. 基础依赖 (Torch 和 Ninja)
    "pip_requirements": [
        # 对应: pip3 install torch torchvision torchaudio
        # 注意: 如果你需要特定的 CUDA 版本(如11.8)，可以在这里加上 --index-url 参数
        "torch torchvision torchaudio", 
        
        # 对应: pip install packaging ninja
        "packaging",
        "ninja"
    ],

    "complex_installs": [
        # 2. 处理 Git Clone 和 项目安装
        # 对应: git clone ...; cd openvla-oft; pip install -e .
        {
            "custom_cmd": [
                # === [修改点] ===
                # 意思是：如果 openvla-oft 目录不存在，才执行 clone；否则打印提示并跳过
                # test -d 检查目录是否存在; || 表示"或者"(前面的失败了才执行后面的)
                "test -d openvla && echo '[INFO] Repo already exists, skipping clone.' || git clone https://github.com/openvla/openvla.git",                
                # "git clone https://github.com/moojink/openvla-oft.git",
                
                # 【技巧】不用在 Python 里执行 cd，直接指定目录路径安装即可
                # 这样比 os.chdir 更安全，不容易弄乱脚本的执行路径
                "pip install -e openvla" 
            ]
        },

        # 3. 安装 Flash Attention
        # 对应: ninja --version; pip install "flash-attn==2.5.5" ...
        {
            "custom_cmd": [
                # 先验证 ninja (如果这步报错，脚本会自动停止，起到了 verify 的作用)
                "ninja --version", 
                
                # 建议先清理缓存(脚本注释中提到的)，防止安装旧的坏包
                # "pip cache remove flash_attn", 
                
                # "pip install flash-attn==2.5.5 --no-build-isolation"
                # "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
                "pip install /mnt/hwfile/tangyuhang/tyh2/whl/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
            ]
        }
    ],
    
    # 4. Git 子模块 (原脚本没提，但如果有需要可以开启)
    "init_submodules": False, 
}

# ==========================================
# 自动化执行逻辑 (通常无需修改)
# ==========================================

def run_cmd(cmd, env=None, cwd=None, shell=True):
    """执行 Shell 命令并实时打印输出"""
    print(f"\n[EXEC] {cmd}")
    
    # 合并环境变量
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
        print(f"[ENV] {env}")

    try:
        # 使用 shell=True 允许复杂的 shell 命令，但要注意安全（本地脚本通常没问题）
        process = subprocess.Popen(
            cmd, 
            shell=shell, 
            cwd=cwd, 
            env=run_env,
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    except Exception as e:
        print(f"[ERROR] Command failed: {e}")
        sys.exit(1)

def check_uv():
    """检查是否安装了 uv"""
    if shutil.which("uv") is None:
        print("[INFO] uv not found, installing...")
        run_cmd("pip install uv")

def setup_conda(env_name, py_ver):
    """简单的 Conda 环境检查与创建提示"""
    # 注意：Python 脚本很难直接激活 Conda 环境并继续运行自身
    # 通常做法是：脚本只负责生成命令，或者用户需要在已有环境内运行此脚本
    print(f"\n=== Environment Check: {env_name} ===")
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    
    if current_env != env_name:
        print(f"[WARNING] You are currently in '{current_env}', but config targets '{env_name}'.")
        print(f"建议先手动运行: conda create -n {env_name} python={py_ver} -y && conda activate {env_name}")
        cont = input("是否继续在当前环境安装? (y/n): ")
        if cont.lower() != 'y':
            sys.exit(0)
    else:
        print(f"[OK] Active environment matches target: {env_name}")

def main():

    # === [新增] 解决 "dubious ownership" 报错 ===
    # 强制 Git 信任所有目录(尤其是 /tmp 下的临时构建目录)
    # 这里的 check=False 是为了防止在没有 git 的机器上报错(虽然不太可能)
    try:
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"], check=False)
        print("[INFO] Applied git safe.directory fix.")
    except Exception:
        pass
    # ===========================================

    # === [新增] 全局环境变量注入 ===
    # 这会让随后所有的 pip / git / uv 操作都自动继承这些设置
    if "global_env" in CONFIG:
        os.environ.update(CONFIG["global_env"])
        print(f"[INFO] Global Env Vars Set: {CONFIG['global_env']}")
    # ============================

    # 1. 环境检查
    setup_conda(CONFIG["project_name"], CONFIG["python_version"])
    
    # 2. 处理 Git 子模块
    if CONFIG.get("init_submodules"):
        print("\n=== Initializing Git Submodules ===")
        run_cmd("git submodule update --init --recursive")

    # 3. UV 设置 (如果启用)
    if CONFIG.get("use_uv"):
        check_uv()
        print("\n=== Running UV Sync ===")
        # 示例：针对 openpi 的特殊环境变量
        uv_env = {"GIT_LFS_SKIP_SMUDGE": "1"} if "openpi" in CONFIG["project_name"] else {}
        run_cmd("uv sync", env=uv_env)

    # 4. 安装基础依赖
    if CONFIG.get("pip_requirements"):
        print("\n=== Installing Basic Requirements ===")
        for req in CONFIG["pip_requirements"]:
            if req.endswith(".txt"):
                run_cmd(f"pip install -r {req}")
            else:
                run_cmd(f"pip install {req}")

    # 5. 执行复杂安装 (核心部分)
    if CONFIG.get("complex_installs"):
        print("\n=== Running Complex Installations ===")
        for item in CONFIG["complex_installs"]:
            
            # 情况 A: 自定义命令列表 (处理像 git clone 这种非 pip 操作)
            if "custom_cmd" in item:
                for cmd in item["custom_cmd"]:
                    run_cmd(cmd, env=item.get("env"))
                continue

            # 情况 B: pip 安装
            pkg = item.get("package")
            flags = " ".join(item.get("flags", []))
            
            # 构建 pip 命令
            # 如果使用了 uv，可以尝试用 uv pip install (视情况而定)
            base_cmd = "uv pip install" if CONFIG.get("use_uv") else "pip install"
            cmd_str = f"{base_cmd} {flags} {pkg}"
            
            run_cmd(cmd_str, env=item.get("env"))

    # 6. 后置处理
    if CONFIG.get("post_setup_cmds"):
        print("\n=== Running Post-Setup Commands ===")
        for cmd in CONFIG["post_setup_cmds"]:
            run_cmd(cmd)

    print("\n✅ Environment Setup Complete!")

if __name__ == "__main__":
    main()