# 以下三个配置还没有验证

# 场景 1: Wall-X (复杂编译，混合 Git 安装)
# 这个配置最复杂，因为它有 MAX_JOBS 这种编译参数，还有手动 git checkout。

CONFIG = {
    "project_name": "wallx",
    "python_version": "3.10",
    "use_uv": False,
    
    "pip_requirements": ["requirements.txt"], # 假设目录下有这个文件

    "complex_installs": [
        # 1. 安装 flash-attn (带编译参数)
        {
            "package": "flash-attn==2.7.4.post1",
            "env": {"MAX_JOBS": "4"},
            "flags": ["--no-build-isolation"]
        },
        # 2. 安装 LeRobot (Git clone + checkout + install)
        {
            "custom_cmd": [
                "git clone https://github.com/huggingface/lerobot.git _deps/lerobot",
                "cd _deps/lerobot && git checkout c66cd401767e60baece16e1cf68da2824227e076cd",
                "pip install -e _deps/lerobot"
            ]
        },
        # 3. 安装 Wall-X (当前目录)
        {
            "init_submodules": True, # 先更新子模块
            "package": "-e .",
            "env": {"MAX_JOBS": "4"},
            "flags": ["--no-build-isolation", "--verbose"]
        }
    ]
}

# 场景 2: OpenPi (使用 UV，涉及 Git LFS)
# 这个配置展示了如何利用 use_uv 和环境变量。

CONFIG = {
    "project_name": "openpi_env",
    "python_version": "3.10",
    "use_uv": True, # 开启 uv 模式
    
    "init_submodules": True, # git submodule update...
    
    "post_setup_cmds": [
        # OpenPi 特有的 uv sync 命令，带特殊环境变量
        "GIT_LFS_SKIP_SMUDGE=1 uv sync", 
        "GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ."
    ]
}

# 场景 3: NVIDIA Isaac GR00T (依赖顺序，特定 Flash-attn)
# 这里展示了如何处理 NVIDIA 对 Flash Attention 的版本锁定。

CONFIG = {
    "project_name": "gr00t",
    "python_version": "3.10",
    "use_uv": True,

    "init_submodules": True,

    "post_setup_cmds": [
        "uv sync --python 3.10",
        "uv pip install -e ."
    ],

    # 如果你需要手动处理 CUDA 11.8 的特殊情况，可以在这里加
    "complex_installs": [
        {
            # 仅当检测到 CUDA 11.8 时手动启用此配置(需要在脚本里加逻辑，或者人工切换)
            "package": "flash-attn==2.8.2",
            "flags": ["--no-build-isolation"]
        }
    ]
}