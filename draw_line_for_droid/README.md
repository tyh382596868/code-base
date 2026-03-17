## file介绍
### CalibrationALL.py




## 注意
### 1.部分视频画来的出轨迹与视频上末端执行器不对应
oxe_utils/transforms.py文件里的以下一段代码要注释掉,因为以下这段代码会随机交换两个第三视角，导致画轨迹时，轨迹画在错误的视角上。
```python
# trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
#     rand_swap_exterior_images(
#         trajectory["observation"]["exterior_image_1_left"],
#         trajectory["observation"]["exterior_image_2_left"],
#     )
# )
```

### 2.LeRobotDataset保存轨迹时越保存越慢
lerobot-0.3.3/src/lerobot/datasets/lerobot_dataset.py里的_save_episode_table方法：
concatenate_datasets会不断追加新的数据到self.hf_dataset导致后续越来越慢，给注释掉。加速保存。

```python
def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
    episode_dict = {key: episode_buffer[key] for key in self.hf_features}
    ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
    ep_dataset = embed_images(ep_dataset)
    # self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
    # self.hf_dataset.set_transform(hf_transform_to_torch)
    ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)

```

### 3. 一条轨迹有外参但是没有内参。
episode_id在cam2base_extrinsics的情况下也有可能不在intrinsics，所以不能单纯判断episode_id在不在外参文件里。

```python
# 如果 episode_id 不在 cam2base_extrinsics 或 intrinsics，就跳过 
if episode_id not in cam2base_extrinsics or episode_id not in intrinsics: 
    continue
```

### 4.一条轨迹内外参数都有，但是内外参的相机id对不上
从外参中取到的相机id在内参中没有返回None，外层循环里判断返回的intrinsics_matrix是不是None，如果是None，这次循环就跳过，不保存这段轨迹。

```python
def get_extrinsics_intrinsics_calib_image_name(episode_id, cam2base_extrinsics, intrinsics, camera_serials):
    # Iterate through the extrinsics to find key that is a digit
    # This is the camera serial number, and the corresponding value is the extrinsics

    ......

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
    ......

```

```python
def save_as_lerobot_dataset(lerobot_dataset: LeRobotDataset, raw_dataset: tf.data.Dataset, **kwargs):

        ......

        cam_to_base_extrinsics_matrix, intrinsics_matrix, calib_image_name = get_extrinsics_intrinsics_calib_image_name(episode_id, cam2base_extrinsics, intrinsics, camera_serials)
        
        if intrinsics_matrix is None:
            continue

        ......

```