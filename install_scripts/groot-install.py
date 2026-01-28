import subprocess
import sys
import os
import platform
import shutil

# ==========================================
# 用户配置区域 (修改这里以适配不同项目)
# ==========================================




CONFIG = {
    "project_name": "groot",  
    "python_version": "3.10",
    "use_uv": True,                

    # === [关键修改] 全局环境变量配置 ===
    "global_env": {
        # 1. 配置 pip 镜像源
        "PIP_INDEX_URL": "https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/",
        "PIP_TRUSTED_HOST": "pkg.pjlab.org.cn",

        # === 2. 额外源 (Nvidia) ===
        # 【关键】专门给 uv 看的。
        # 只有当 PJLab 里找不到包(比如 tensorrt-libs)时，uv 才会去这里找
        "UV_EXTRA_INDEX_URL": "https://pypi.nvidia.com",

        # 2. 【核心解决方案】配置不走代理的白名单
        # 告诉 pip 和其他工具：访问 pjlab.org.cn 时，不要经过代理
        # localhost,127.0.0.1 是常规建议，防止本地通讯也走代理出问题
        "no_proxy": "localhost,127.0.0.1,pkg.pjlab.org.cn",
        "NO_PROXY": "localhost,127.0.0.1,pkg.pjlab.org.cn" 
    },

    "post_setup_cmds": [
        # "uv remove flash-attn",

        "uv sync --python 3.10",
        ## 远程安装
        # "uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        ## or 本地安装
        # "uv pip install /mnt/petrelfs/tangyuhang/tyh2/whl/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"


        "uv pip install -e .",
        # 4. [新增] 打印环境信息进行自检
        # 注意：外层用双引号，内层 f-string 用单引号，避免转义地狱
        "uv run python -c \"import torch; import platform; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Python: {platform.python_version()}')\""     
    ],
    
    # 4. Git 子模块
    "init_submodules": True, 
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
        # run_cmd("uv sync", env=uv_env)

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