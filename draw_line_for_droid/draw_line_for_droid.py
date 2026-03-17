import argparse
import re
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from oxe_utils.configs import OXE_DATASET_CONFIGS, ActionEncoding, StateEncoding
from oxe_utils.transforms import OXE_STANDARDIZATION_TRANSFORMS


from openx_rlds import transform_raw_dataset, generate_features_from_raw

from lerobot.datasets.utils import append_jsonlines
import json
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import mediapy
from scipy.spatial.transform import Rotation as R
import cv2
import imageio
import time
np.set_printoptions(precision=2)

def get_cam_info():
    path_to_droid_repo = "/mnt/hwfile/tangyuhang/.cache/huggingface/hub/models--KarlP--droid/snapshots/bcb840c3b496533e0adf548a54b51f2f00057837" # TODO: Replace with the path to your DROID repository

    # Load the extrinsics
    cam2base_extrinsics_path = f"{path_to_droid_repo}/cam2base_extrinsics.json"
    with open(cam2base_extrinsics_path, "r") as f:
        cam2base_extrinsics = json.load(f)

    # Load the intrinsics
    intrinsics_path = f"{path_to_droid_repo}/intrinsics.json"
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)

    # Load mapping from episode ID to path, then invert
    episode_id_to_path_path = f"{path_to_droid_repo}/episode_id_to_path.json"
    with open(episode_id_to_path_path, "r") as f:
        episode_id_to_path = json.load(f)
    episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

    # Load camera serials
    camera_serials_path = f"{path_to_droid_repo}/camera_serials.json"
    with open(camera_serials_path, "r") as f:
        camera_serials = json.load(f)    

    return cam2base_extrinsics, intrinsics, episode_id_to_path, episode_path_to_id, camera_serials

def get_extrinsics_intrinsics_calib_image_name(episode_id, cam2base_extrinsics, intrinsics, camera_serials):
    # Iterate through the extrinsics to find key that is a digit
    # This is the camera serial number, and the corresponding value is the extrinsics
    for k, v in cam2base_extrinsics[episode_id].items():
        if k.isdigit():
            camera_serial = k
            extracted_extrinsics = v
            break
    # # intrinsics 里没有对应 serial（避免 KeyError）
    if camera_serial not in intrinsics[episode_id]:
        SKIP_LOG_PATH = Path("/mnt/petrelfs/tangyuhang/tyh2/code-base/missing_calib.jsonl")  # 你也可以换成绝对路径
        record = {
            "episode_id": episode_id,
            "camera_serial": camera_serial,
            "cam2base_extrinsics_episode": cam2base_extrinsics[episode_id],
            "intrinsics_episode": intrinsics[episode_id],
        }        
        append_jsonlines(record, SKIP_LOG_PATH)

        return None, None, None
    # Also lets us get the intrinsics
    extracted_intrinsics = intrinsics[episode_id][camera_serial]

    # Using the camera serial, find the corresponding camera name (which is used to determine
    # which image stream in the episode to use)
    camera_serials_to_name = {v: k for k, v in camera_serials[episode_id].items()}
    calib_camera_name = camera_serials_to_name[camera_serial]

    if calib_camera_name == "ext1_cam_serial":
        calib_image_name = "exterior_image_1_left"
    elif calib_camera_name == "ext2_cam_serial":
        calib_image_name = "exterior_image_2_left"
    else:
        raise ValueError(f"Unknown camera name: {calib_camera_name}")

    print(f"Camera with calibration data: {calib_camera_name} --> {calib_image_name}")    

    # Convert the extrinsics to a homogeneous transformation matrix
    pos = extracted_extrinsics[0:3] # translation
    rot_mat = R.from_euler("xyz", extracted_extrinsics[3:6]).as_matrix() # rotation

    # Make homogenous transformation matrix
    cam_to_base_extrinsics_matrix = np.eye(4)
    cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
    cam_to_base_extrinsics_matrix[:3, 3] = pos

    print(cam_to_base_extrinsics_matrix)

    # Convert the intrinsics to a matrix
    fx, cx, fy, cy = extracted_intrinsics["cameraMatrix"]
    intrinsics_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
    ])
    print(intrinsics_matrix)

    return cam_to_base_extrinsics_matrix, intrinsics_matrix, calib_image_name

def get_images_ee_poses(ep, calib_image_name):
    # Save all observations for the calibrated camera and corresponding gripper positions
    images = []
    cartesian_poses = []
    eq_len = len(ep['steps']['observation']['cartesian_position'])
    for index in range(eq_len):
        image = ep['steps']['observation'][calib_image_name][index]
        images.append(image)
        cartesian_pose = ep['steps']['observation']['cartesian_position'][index]
        cartesian_poses.append(cartesian_pose)

    return images, cartesian_poses

def get_pixel_positions(cartesian_poses, cam_to_base_extrinsics_matrix, intrinsics_matrix):
    # length images x 6
    cartesian_poses = np.array(cartesian_poses)
    # Remove the rotation and make homogeneous: --> length images x 3 --> length images x 4
    cartesian_homogeneous_positions = cartesian_poses[:, :3]
    cartesian_homogeneous_positions = np.hstack(
        (cartesian_homogeneous_positions, np.ones((cartesian_homogeneous_positions.shape[0], 1)))
    )

    # Transpose to support matrix multiplication: --> 4 x length images
    gripper_position_base = cartesian_homogeneous_positions.T    

    # Transform gripper position to camera frame, then remove homogeneous component
    base_to_cam_extrinsics_matrix = np.linalg.inv(cam_to_base_extrinsics_matrix)
    robot_gripper_position_cam = base_to_cam_extrinsics_matrix @ gripper_position_base
    robot_gripper_position_cam = robot_gripper_position_cam[:3] # Now 3 x length images

    # Finally, use intrinsics to project the gripper position in camera frame into pixel space
    pixel_positions = intrinsics_matrix @ robot_gripper_position_cam[:3]
    pixel_positions = pixel_positions[:2] / pixel_positions[2]

    return pixel_positions


def save_video(images, pixel_positions, episode_id, path_to_droid_repo):

    # 可视化并保存视频
    vis_images = []
    temp_img_path = f"{path_to_droid_repo}/temp/TEMP.png"

    for i, image in enumerate(tqdm(images)):
        if i % 10 != 0:
            continue
        
        fig, axs = plt.subplots(1)
        x, y = pixel_positions[0, i] / 1280 * 320, pixel_positions[1, i] / 720 * 180  # Scale to match image dimensions

        # clip coords
        x = np.clip(x, 0, 320)
        y = np.clip(y, 0, 180)

        axs.imshow(image)
        axs.scatter(x, y, c='red', s=20)
        axs.set_xlim(0, 320)
        axs.set_ylim(180, 0)  # Invert y-axis to match image

        # turn off axes
        axs.axis('off')

        # save the figure, then reopen it as PIL image
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        vis_image = Image.open(temp_img_path).convert("RGB")
        vis_images.append(np.array(vis_image))

    # 保存为 mp4 视频
    output_video_path = f"{path_to_droid_repo}/video/{episode_id}.mp4"
    imageio.mimsave(output_video_path, vis_images, fps=8)
    print(f"视频已保存到: {output_video_path}")

def draw_25d(background_image,pixel_positions, index, draw_len=8):
    
    # length of seq
    episode_len = len(pixel_positions[0])

    # Create a new black image to draw the trajectory on.
    # image=np.zeros((img_height,img_width,3),dtype=np.uint8)
    # demonstration背景
    image = background_image.copy()

    # Store 2D positions of the eef in this episode in a list
    TempProgress = []

    # Store gripper height in a list.
    gripper_height_list = []

    start = index

    if start+draw_len <= episode_len:
        end = start+draw_len
    else:
        end = episode_len

    for i in range(start,end):

        
        u,v = pixel_positions[0, i] / 1280 * 320, pixel_positions[1, i] / 720 * 180 # Scale to match image dimensions

        TempProgress.append((int(u), int(v)))

        # Store the gripper height in a list.
        gripper_height = v
        gripper_height_list.append(gripper_height)

    # Draw gripper height in green color. The higher, the lighter.
    # Draw the Temporal Progress in red color. The earlier the time, the lighter the color of the line segment. Line thickness is 3.
    max_height = max(gripper_height_list)
    min_height = min(gripper_height_list)

    for i in range(1, len(gripper_height_list)):
        # Normalize the gripper height to [0,1]
        if min_height != max_height:
            normalized_gripper_height = float(gripper_height_list[i] - min_height) / (max_height - min_height)
            color = (200, int(255 * normalized_gripper_height), 255 * i / (episode_len - 1))
        else:
            color = (200, 255, 255 * i / (episode_len - 1))
        cv2.line(image, TempProgress[i - 1], TempProgress[i], color=color, thickness=8)

    return image



def save_as_lerobot_dataset(lerobot_dataset: LeRobotDataset, raw_dataset: tf.data.Dataset, **kwargs):
    cam2base_extrinsics, intrinsics, episode_id_to_path, episode_path_to_id, camera_serials = get_cam_info()
    resume_index=0
    resume_checkpoint=12600+7770
    for episode in tqdm(raw_dataset.as_numpy_iterator()):
        start_time = time.perf_counter()

        file_path = episode["episode_metadata"]["file_path"].decode("utf-8")
        recording_folderpath = episode["episode_metadata"]["recording_folderpath"].decode("utf-8")

        episode_path = file_path.split("r2d2-data-full/")[1].split("/trajectory")[0]
        if episode_path not in episode_path_to_id:
            continue
        episode_id = episode_path_to_id[episode_path]
        
        # 如果 episode_id 不在 cam2base_extrinsics 或 intrinsics，就跳过 
        if episode_id not in cam2base_extrinsics or episode_id not in intrinsics: 
            continue

        resume_index+=1
        if resume_index<=resume_checkpoint:
            print(f"Less than resume:{resume_index}<={resume_checkpoint}")
            continue

        # breakpoint()  
        cam_to_base_extrinsics_matrix, intrinsics_matrix, calib_image_name = get_extrinsics_intrinsics_calib_image_name(episode_id, cam2base_extrinsics, intrinsics, camera_serials)
        
        if intrinsics_matrix is None:
            continue
        images, cartesian_poses = get_images_ee_poses(episode, calib_image_name)
        pixel_positions = get_pixel_positions(cartesian_poses, cam_to_base_extrinsics_matrix, intrinsics_matrix)

        # save_video(images, pixel_positions, episode_id, "/mnt/petrelfs/tangyuhang/droid/droid_lerobot/demo")

        traj = episode["steps"]
        for i in range(traj["action"].shape[0]):

            image_dict = {
                f"observation.images.{key}": value[i]
                for key, value in traj["observation"].items()
                if "depth" not in key and any(x in key for x in ["image", "rgb"])
            }

            traj_image = draw_25d(images[i],pixel_positions, i, draw_len=30)
            image_dict["observation.images.image_traj"] = traj_image


            lerobot_dataset.add_frame(
                {
                    **image_dict,
                    "observation.state": traj["proprio"][i],
                    "action": traj["action"][i],
                },
                task=traj["task"][0].decode(),
            )
            end_time = time.perf_counter()
            print(f"############ time of saving an episode: {end_time - start_time}")
        lerobot_dataset.save_episode()


def create_lerobot_dataset(
    raw_dir: Path,
    repo_id: str = None,
    local_dir: Path = None,
    push_to_hub: bool = False,
    fps: int = None,
    robot_type: str = None,
    use_videos: bool = True,
    image_writer_process: int = 5,
    image_writer_threads: int = 10,
    keep_images: bool = True,
):
    last_part = raw_dir.name
    if re.match(r"^\d+\.\d+\.\d+$", last_part):
        version = last_part
        dataset_name = raw_dir.parent.name
        data_dir = raw_dir.parent.parent
    else:
        version = ""
        dataset_name = last_part
        data_dir = raw_dir.parent

    if local_dir is None:
        local_dir = Path(HF_LEROBOT_HOME)
    local_dir /= f"{dataset_name}_{version}_lerobot"
    if local_dir.exists():
        shutil.rmtree(local_dir)

    builder = tfds.builder(dataset_name, data_dir=data_dir, version=version)
    features = generate_features_from_raw(builder, use_videos)
    filter_fn = lambda e: e["success"] if dataset_name == "kuka" else True
    raw_dataset = (
        builder.as_dataset(split="train")
        .filter(filter_fn)
        .map(partial(transform_raw_dataset, dataset_name=dataset_name))
    )

    if fps is None:
        if dataset_name in OXE_DATASET_CONFIGS:
            fps = OXE_DATASET_CONFIGS[dataset_name]["control_frequency"]
        else:
            fps = 10

    if robot_type is None:
        if dataset_name in OXE_DATASET_CONFIGS:
            robot_type = OXE_DATASET_CONFIGS[dataset_name]["robot_type"]
            robot_type = robot_type.lower().replace(" ", "_").replace("-", "_")
        else:
            robot_type = "unknown"

    features["observation.images.image_traj"] = features["observation.images.wrist_image_left"]

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        root=local_dir,
        fps=int(fps),
        use_videos=use_videos,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_process,
    )

    save_as_lerobot_dataset(lerobot_dataset, raw_dataset, keep_images=keep_images)

    if push_to_hub:
        assert repo_id is not None
        tags = ["LeRobot", dataset_name, "rlds"]
        if dataset_name in OXE_DATASET_CONFIGS:
            tags.append("openx")
        if robot_type != "unknown":
            tags.append(robot_type)
        lerobot_dataset.push_to_hub(
            tags=tags,
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `path/to/dataset` or `path/to/dataset/version).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to hub.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=None,
        help="Robot type of this dataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frame rate used to collect videos. Default fps equals to the control frequency of the robot.",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
    )
    parser.add_argument(
        "--image-writer-process",
        type=int,
        default=5,
        help="Number of processes of image writer for saving images.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="Number of threads per process of image writer for saving images.",
    )

    args = parser.parse_args()
    create_lerobot_dataset(**vars(args))


if __name__ == "__main__":
    main()