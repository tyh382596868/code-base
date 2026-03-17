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
from lerobot.datasets.utils import append_jsonlines
np.set_printoptions(precision=2)


import json
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import mediapy
from scipy.spatial.transform import Rotation as R
import cv2
import imageio

def save_video(images, pixel_positions, episode_id, path_to_droid_repo):

    # 可视化并保存视频
    vis_images = []
    temp_img_path = f"{path_to_droid_repo}/temp/TEMP.png"

    for i, image in enumerate((images)):
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

def main():
    # 指定本地缓存目录
    ds = tfds.load(
        "droid",                # 数据集名称
        split="train",          # 数据集划分
        data_dir="/mnt/hwfile/tangyuhang"  # 本地缓存路径
    )

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
    # Iterate through the dataset to find the first episode that has a cam2base extrinsics entry
    eps = []
    for ep in tqdm(ds):
        file_path = ep["episode_metadata"]["file_path"].numpy().decode("utf-8")
        recording_folderpath = ep["episode_metadata"]["recording_folderpath"].numpy().decode("utf-8")

        episode_path = file_path.split("r2d2-data-full/")[1].split("/trajectory")[0]
        if episode_path not in episode_path_to_id:
            continue
        episode_id = episode_path_to_id[episode_path]

        
        if (episode_id not in cam2base_extrinsics) or (episode_id not in intrinsics):
            continue

        # Iterate through the extrinsics to find key that is a digit
        # This is the camera serial number, and the corresponding value is the extrinsics
        for k, v in cam2base_extrinsics[episode_id].items():
            if k.isdigit():
                camera_serial = k
                extracted_extrinsics = v
                break

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

        # Save all observations for the calibrated camera and corresponding gripper positions
        images = []
        cartesian_poses = []
        for step in ep["steps"]:
            image = step["observation"][calib_image_name].numpy()
            images.append(image)
            cartesian_pose = step["observation"]["cartesian_position"].numpy()
            cartesian_poses.append(cartesian_pose)

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

        save_video(images, pixel_positions, episode_id, "/mnt/petrelfs/tangyuhang/droid/droid_lerobot/demo2")


if __name__=="__main__":
        main()