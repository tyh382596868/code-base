import json
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import mediapy
import imageio
from scipy.spatial.transform import Rotation as R

def draw_line(episode_index,cam2base_extrinsics,intrinsics,episode_path_to_id,camera_serials,episode_id,ep):

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
    append_jsonlines({episode_index:calib_camera_name},Path("/mnt/hwfile/tangyuhang/droid/droid_lerobot/droid_1.0.0_lerobot/meta/idx_to_calib_camera_name.jsonl"))


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


    # Visualize!
    vis_images = []
    temp_img_path = f"{path_to_droid_repo}/TEMP.png"

    for i, image in enumerate(tqdm(images)):
        if i % 10 != 0:
            continue
        
        fig, axs = plt.subplots(1)
        x, y = pixel_positions[0, i] / 1280 * 320, pixel_positions[1, i] / 720 * 180 # Scale to match image dimensions

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

    mediapy.write_video(f'episode_{episode_index:06d}.mp4', vis_images, fps=15)




