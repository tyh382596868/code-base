import argparse
import re
import shutil
from functools import partial
from pathlib import Path
import imageio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_jsonlines
from datasets import Dataset
np.set_printoptions(precision=2)

features= {
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": [
            256,
            256,
            3
        ],
        "names": [
            "height",
            "width",
            "rgb"
        ],
    },
    "observation.images.image": {
        "dtype": "video",
        "shape": [
            256,
            256,
            3
        ],
        "names": [
            "height",
            "width",
            "rgb"
        ],
    },
    "observation.images.image_traj": {
        "dtype": "video",
        "shape": [
            256,
            256,
            3
        ],
        "names": [
            "height",
            "width",
            "rgb"
        ],
    },    
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "motors": [
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
                "gripper"
            ]
        }
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "motors": [
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper"
            ]
        }
    },
}

def get_video(video_path):
    # read video
    reader = imageio.get_reader(video_path)
    # 获取视频的元信息
    print("帧率:", reader.get_meta_data()['fps'])
    print("总帧数:", reader.count_frames())

    imgs = []
    for i, frame in enumerate(reader):
        # print("Frame", i, frame.shape)  # frame 是一个 numpy 数组
        imgs.append(np.array(frame))
    reader.close()    
    return imgs


def save_as_lerobot_dataset(lerobot_dataset: LeRobotDataset, libero_parquet_dir: Path):
    # breakpoint()
    task_index_to_name = load_jsonlines(str(libero_parquet_dir.parent.parent / "meta" / "tasks.jsonl"))
    for parquet_file in libero_parquet_dir.iterdir():

        ### get image
        video1_path = libero_parquet_dir.parent.parent / "videos/chunk-000" / "observation.images.image" / f"{parquet_file.stem}.mp4"
        video2_path = libero_parquet_dir.parent.parent / "videos/chunk-000" / "observation.images.wrist_image" / f"{parquet_file.stem}.mp4"
        video3_path = libero_parquet_dir.parent.parent / "videos_traj/chunk-000" / "observation.images.image" / f"{parquet_file.stem}.mp4"

        video1 = get_video(video1_path)
        video2 = get_video(video2_path)
        video3 = get_video(video3_path)


        traj = Dataset.from_parquet(str(parquet_file))
        for i in range(len(traj["action"])):
            image_dict = {
                "observation.images.image": video1[i],
                "observation.images.wrist_image": video2[i],
                "observation.images.image_traj": video3[i],
            }     

            lerobot_dataset.add_frame(
                {   **image_dict,
                    "observation.state": np.array(traj["observation.state"][i], dtype=np.float32),
                    "action": np.array(traj["action"][i], dtype=np.float32),
                },
                task=task_index_to_name[traj["task_index"][0]]["task"],
            )
        lerobot_dataset.save_episode()


def create_lerobot_dataset(
    raw_dir: Path,
    repo_id: str = None,
    local_dir: Path = None,
):
    
    local_dir /= raw_dir.name
    if local_dir.exists():
        shutil.rmtree(local_dir)

    fps = 20
    robot_type = "franka"

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        root=local_dir,
        fps=int(fps),
        features=features,
    )

    libero_parquet_dir = raw_dir / "data/chunk-000"
    save_as_lerobot_dataset(lerobot_dataset, libero_parquet_dir)




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
    args = parser.parse_args()
    create_lerobot_dataset(**vars(args))


if __name__ == "__main__":
    main()