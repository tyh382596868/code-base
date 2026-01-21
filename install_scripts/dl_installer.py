import subprocess
import sys
import os
import platform
import shutil

# ==========================================
# 用户配置区域 (修改这里以适配不同项目)
# ==========================================

CONFIG = {
    "project_name": "wall-x-env",  # Conda 环境名称
    "python_version": "3.10",
    "use_uv": False,               # 是否使用 uv (如 openpi/GR00T)

    # === [新增] 全局镜像源配置 ===
    # 只要配在这里，下面所有的 pip install 都不用再写 -i 参数了
    "global_env": {
        "PIP_INDEX_URL": "https://pkg.pjlab.org.cn/repository/pypi-proxy/simple/",
        "PIP_TRUSTED_HOST": "pkg.pjlab.org.cn"
    },

    # 1. 基础依赖安装 (pip install ...)
    "pip_requirements": [
        # 可以是文件路径，也可以是具体的包名
        # "requirements.txt", 
    ],

    # 2. 复杂的 pip 安装 (支持环境变量和特殊 flag)
    # 格式: {"package": "包名/路径", "env": {环境变量}, "flags": [参数列表]}
    "complex_installs": [
        {
            # 对应 Wall-X 的 flash-attn 安装
            "package": "flash-attn==2.7.4.post1",
            "env": {"MAX_JOBS": "4"},
            "flags": ["--no-build-isolation"]
        },
        {
            # 对应 Wall-X 的 lerobot 安装 (Git 源码安装)
            "package": ".", # 先 clone 再安装，这里假设已经在目录内，或者后续手动指定
            "custom_cmd": [
                # 如果需要从 git 安装特定 commit，可以在这里定义预处理命令
                "git clone https://github.com/huggingface/lerobot.git _deps/lerobot",
                "cd _deps/lerobot && git checkout c66cd401767e60baece16e1cf68da2824227e076cd",
                "pip install -e _deps/lerobot"
            ]
        },
        {
            # 对应 Wall-X 的主包安装
            "package": "-e .",
            "env": {"MAX_JOBS": "4"},
            "flags": ["--no-build-isolation", "--verbose"]
        }
    ],

    # 3. Git 子模块处理 (对应 Wall-X/OpenPi 的 submodule update)
    "init_submodules": True,  # 执行 git submodule update --init --recursive
    
    # 4. 其他 Shell 命令 (如 openpi 的 GIT_LFS_SKIP_SMUDGE)
    "post_setup_cmds": [
        # "GIT_LFS_SKIP_SMUDGE=1 uv sync"
    ]
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
    if CONFIG["pip_requirements"]:
        print("\n=== Installing Basic Requirements ===")
        for req in CONFIG["pip_requirements"]:
            if req.endswith(".txt"):
                run_cmd(f"pip install -r {req}")
            else:
                run_cmd(f"pip install {req}")

    # 5. 执行复杂安装 (核心部分)
    if CONFIG["complex_installs"]:
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