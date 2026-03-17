import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from lerobot.datasets.compute_stats import aggregate_stats, auto_downsample_height_width, get_feature_stats, sample_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    load_json,
    load_jsonlines
)
from lerobot.datasets.video_utils import get_safe_default_codec

from datasets import Dataset
import tqdm
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

features= {
"observation.images.wrist_image_left": {
    "dtype": "video",
    "shape": [
        180,
        320,
        3
    ],
    "names": [
        "height",
        "width",
        "rgb"
    ]
},
"observation.images.exterior_image_1_left": {
    "dtype": "video",
    "shape": [
        180,
        320,
        3
    ],
    "names": [
        "height",
        "width",
        "rgb"
    ]
},
"observation.images.exterior_image_2_left": {
    "dtype": "video",
    "shape": [
        180,
        320,
        3
    ],
    "names": [
        "height",
        "width",
        "rgb"
    ]
},
"observation.images.image_traj": {
    "dtype": "video",
    "shape": [
        180,
        320,
        3
    ],
    "names": [
        "height",
        "width",
        "rgb"
    ]
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
            "pad",
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




def sample_images(input):
    if type(input) is str:
        video_path = input
        reader = torchvision.io.VideoReader(video_path, stream="video")
        frames = [frame["data"] for frame in reader]
        frames_array = torch.stack(frames).numpy()  # Shape: [T, C, H, W]

        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

            images[i] = img
    elif type(input) is np.ndarray:
        frames_array = input[:, None, :, :]  # Shape: [T, C, H, W]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)

            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

            images[i] = img

    return images


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array
        breakpoint()
        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        if features[key]["dtype"] in ["image", "video"]:
            value_norm = 1.0 if "depth" in key else 255.0
            ep_stats[key] = {k: v if k == "count" else np.squeeze(v / value_norm, axis=0) for k, v in ep_stats[key].items()}

    return ep_stats


def invert_gripper_actions(actions: np.ndarray) -> np.ndarray:
    return 1 - actions


class FrctalDatasetMetadata(LeRobotDatasetMetadata):
    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


class FractalDataset(LeRobotDataset):
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        # breakpoint()
        obj.meta = FrctalDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # breakpoint()
        # breakpoint()
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        features = {key: value for key, value in self.features.items() if key in self.hf_features}  # remove video keys
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key, value in frame.items():
            if key not in self.features:
                raise ValueError(f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'.")

            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, videos: dict, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
        
        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)
        
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        self._save_episode_table(episode_buffer, episode_index)
        # breakpoint()
        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

def main(
    repo_id:str,
    src_path: Path,
    output_path: Path,
    fps: int,
    robot_type: str,
    part_paths: list
):
    # breakpoint()
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset: FractalDataset = FractalDataset.create(
        repo_id=f"fratal/{output_path.name}",
        root=output_path,
        fps=fps,
        robot_type=robot_type,
        features=features,
    )

    logging.info(f"start processing for {src_path}, saving to {output_path}")

    start_time = time.perf_counter()

    for part_path in part_paths:
        part_path = src_path / part_path

        info_path = part_path / "meta/info.json"
        meta_info = load_json(info_path)
        num_ep = meta_info["total_episodes"]

        task_index_to_task_path = part_path / "meta/tasks.jsonl"
        task_index_to_tasks = load_jsonlines(task_index_to_task_path)

        for index_ep in range(num_ep):
            num_chunk = index_ep // 1000

            if index_ep <= 5111:
                
                continue
                
            print(f"index_ep:{index_ep}:{part_path}")
            parquet_dir_path = part_path / f"data/chunk-{num_chunk:03d}"
            video_dir_path = part_path / f"videos/chunk-{num_chunk:03d}"

            parquet_path = parquet_dir_path / f"episode_{index_ep:06d}.parquet"
            videos = {
                "observation.images.wrist_image_left": video_dir_path / "observation.images.wrist_image_left" / f"episode_{index_ep:06d}.mp4",
                "observation.images.exterior_image_1_left": video_dir_path / "observation.images.exterior_image_1_left" / f"episode_{index_ep:06d}.mp4",
                "observation.images.exterior_image_2_left": video_dir_path / "observation.images.exterior_image_2_left" / f"episode_{index_ep:06d}.mp4",
                "observation.images.image_traj": video_dir_path / "observation.images.image_traj" / f"episode_{index_ep:06d}.mp4",
            }


            traj = Dataset.from_parquet(str(parquet_path))

            instruction = task_index_to_tasks[traj["task_index"][0]]["task"]

            for i in range(len(traj["action"])):
                dataset.add_frame(
                    {
                        "action": np.array(traj["action"][i], dtype=np.float32),
                        "observation.state": np.array(traj["observation.state"][i], dtype=np.float32),
                    },
                    task=instruction,
                )
            dataset.save_episode(videos=videos)

    end_time = time.perf_counter()
    run_time = end_time - start_time
    print(f"代码运行时间: {run_time:.8f} 秒")

    if dataset.meta.total_episodes == 0:
        shutil.rmtree(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--robot-type", type=str, required=True)

    args = parser.parse_args()

    part_paths = [
        "droid_lerobot/droid_1.0.0_lerobot",
        "droid_lerobot_multi_process/droid_1.0.0_lerobot",
        "droid_lerobot_multi_process_12600+7770/droid_1.0.0_lerobot"
    ]

    main(**vars(args),part_paths=part_paths)