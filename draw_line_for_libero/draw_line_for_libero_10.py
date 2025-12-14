from pathlib import Path
import pathlib
import tqdm
import imageio
from datasets import Dataset

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import camera_utils as CU
import cv2
import numpy as np


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": None,
        "renderer_config": {"offscreen": True},
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    env.reset()
    return env, task_description
    
def projection(x_w,e_matrix,i_matrix):
    # 世界坐标 -> 相机坐标
    x_c = e_matrix[:3,:3].T @ (x_w - e_matrix[:3,3])
    
    
    # 相机坐标 -> 归一化平面
    x_norm = x_c[0] / x_c[2]
    y_norm = x_c[1] / x_c[2]

    fx, cx = i_matrix[0,0], i_matrix[0,2]
    fy, cy = i_matrix[1,1], i_matrix[1,2]
    
    # 投影到像素坐标
    u = fx * x_norm + cx
    v = fy * (-y_norm) + cy

    return u,v
    

def draw_25d(background_image,parquet_data,index):
    # for episode_idx, episode in enumerate(dataset):
    episode_len = len(parquet_data['observation.state'])

    # Create a new black image to draw the trajectory on.
    # image=np.zeros((img_height,img_width,3),dtype=np.uint8)
    # demonstration背景
    image = background_image.copy()

    # Get the total number of steps in this episode

    # print("Total number of episodes in the dataset: ",episode_len)

    # Store 2D positions of the eef in this episode in a list
    TempProgress = []
    # Ensure we only draw a circle as soon as the gripper is closed or opened.
    is_catched = False
    # Store gripper height in a list.
    gripper_height_list = []

    start = index

    if start+32 <= episode_len:
        end = start+32
    else:
        end = episode_len

    for i in range(start,end):

        xyz = parquet_data['observation.state'][i][:3]
        u,v = projection(xyz, agent_ex, intrinsic_matrix)

        TempProgress.append((int(256-u), int(240-v)))

        gripper_closed = (parquet_data['observation.state'][i][-2] - parquet_data['observation.state'][i][-1])
        # print(gripper_closed)
        # Draw Interaction Markers.
        if is_catched == False and gripper_closed < 0.01:
            is_catched = True
            # print(i, is_catched, gripper_closed)
            # If the gripper is closed, draw a green circle on the image.
            cv2.circle(image, (int(256-u), int(240-v)), radius=5, color=(0, 255, 0), thickness=2)
        if is_catched == True and gripper_closed > 0.02:
            is_catched = False
            # print(i, is_catched, gripper_closed)
            # If the gripper is opened, draw a blue circle on the image.
            cv2.circle(image, (int(256-u), int(240-v)), radius=5, color=(255, 0, 0), thickness=2)

        # Store the gripper height in a list.
        gripper_height = 256 - v
        gripper_height_list.append(gripper_height)

    # Draw gripper height in green color. The higher, the lighter.
    # Draw the Temporal Progress in red color. The earlier the time, the lighter the color of the line segment. Line thickness is 3.
    max_height = max(gripper_height_list)
    min_height = min(gripper_height_list)

    for i in range(1, len(gripper_height_list)):
        # Normalize the gripper height to [0,1]
        if min_height != max_height:
            normalized_gripper_height = float(gripper_height_list[i] - min_height) / (max_height - min_height)
            color = (0, int(255 * normalized_gripper_height), 255 * i / (episode_len - 1))
        else:
            color = (0, 255, 255 * i / (episode_len - 1))
        cv2.line(image, TempProgress[i - 1], TempProgress[i], color=color, thickness=2)

    return image


import argparse




LIBERO_ENV_RESOLUTION = 256
seed = 7
task_suite_name = "libero_10"

# Initialize LIBERO task suite
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[task_suite_name]()
num_tasks_in_suite = task_suite.n_tasks

intrinsic_matrix_list = []
agent_ex_list = []
lang_map_index = {}

for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    # Get task
    task = task_suite.get_task(task_id)
    env, language = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
    intrinsic_matrix = CU.get_camera_intrinsic_matrix(
        env.sim, "agentview", LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION
    )
    agent_ex = CU.get_camera_extrinsic_matrix(env.sim, "agentview")
    print("================================================")
    print(f"TASK: {language}\nIntrinsic: {intrinsic_matrix}\nExt: {agent_ex}") 

    intrinsic_matrix_list.append(intrinsic_matrix)
    agent_ex_list.append(agent_ex)
    
    # 添加 language -> task_id 映射
    lang_map_index[language] = task_id

task_index_to_lang = {}

import json
file_path = "/mnt/inspurfs/evla2_t/vla_next_next/data_process/datasets/libero_10_no_noops_1.0.0_lerobot/meta/tasks.jsonl"
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        task = json.loads(line)   # 每一行是一个字典
        task_index_to_lang[task["task_index"]] = task["task"]
        # print(task)



root_path = Path("/mnt/inspurfs/evla2_t/vla_next_next/data_process/datasets")
data_name = f"{task_suite_name}_no_noops_1.0.0_lerobot"

data_path = root_path / data_name

video_path = data_path / "videos/chunk-000/observation.images.image"
parquet_dir_path = data_path / "data/chunk-000"

save_dir =  data_path / "videos_traj/chunk-000/observation.images.image"
if not save_dir.exists():
    save_dir.mkdir(parents=True,exist_ok=True)
    print("原路径不存在，以创建")
    print(save_dir)

for file in tqdm.tqdm(video_path.iterdir()):
    print(file)  # 输出所有 mp4 文件路径

    # read video
    reader = imageio.get_reader(file)
    # 获取视频的元信息
    print("帧率:", reader.get_meta_data()['fps'])
    print("总帧数:", reader.count_frames())

    imgs = []
    for i, frame in enumerate(reader):
        # print("Frame", i, frame.shape)  # frame 是一个 numpy 数组
        imgs.append(np.array(frame))
    reader.close()    

    parquet_path = parquet_dir_path / f"{file.stem}.parquet"
    parquet_data = Dataset.from_parquet(str(parquet_path))

    lang = task_index_to_lang[parquet_data["task_index"][0]]

    index = lang_map_index[lang]

    intrinsic_matrix = intrinsic_matrix_list[index]
    agent_ex = agent_ex_list[index]


    if len(parquet_data["observation.state"])!=len(imgs):
        print(len(parquet_data["observation.state"]))
        print(len(imgs))
        raise ValueError("lenth of state must equal to lenth of frame")

    
    frames = []
    for i in tqdm.tqdm(range(len(imgs))):

        xyz = parquet_data['observation.state'][i][:3]
        u,v = projection(xyz, agent_ex, intrinsic_matrix)

        img = imgs[i]
        draw_img = draw_25d(img,parquet_data,i)

        frames.append(draw_img)
    
    save_path = save_dir / file.name
    print(save_path)
    imageio.mimsave(
        save_path,
        frames,
        fps=30,
        codec="libx264"
    )




