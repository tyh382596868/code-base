import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import numpy as np
# 指定本地缓存目录
ds = tfds.load(
    "droid",                # 数据集名称
    split="train",          # 数据集划分
    data_dir="/mnt/petrelfs/tangyuhang/droid_100"  # 本地缓存路径
)

wrist_image_lefts = []
exterior_image_1_lefts = []
exterior_image_2_lefts = []
joint_positions = []
gripper_positions = []


for ep in tqdm(ds):
    for step in ep["steps"]:
        print(step["language_instruction"])
        wrist_image_left = step["observation"]["wrist_image_left"].numpy()
        exterior_image_1_left = step["observation"]["exterior_image_1_left"].numpy()
        exterior_image_2_left = step["observation"]["exterior_image_2_left"].numpy()
        joint_position = step["observation"]["joint_position"].numpy()
        gripper_position = step["observation"]["gripper_position"].numpy()

        wrist_image_lefts.append(wrist_image_left)
        exterior_image_1_lefts.append(exterior_image_1_left)
        exterior_image_2_lefts.append(exterior_image_2_left)
        joint_positions.append(joint_position)
        gripper_positions.append(gripper_position)



    np.save("wrist_image_lefts.npy", wrist_image_lefts)
    np.save("exterior_image_1_lefts.npy", exterior_image_1_lefts)
    np.save("exterior_image_2_lefts.npy", exterior_image_2_lefts)
    np.save("joint_positions.npy", joint_positions)
    np.save("gripper_positions.npy", gripper_positions)




    break