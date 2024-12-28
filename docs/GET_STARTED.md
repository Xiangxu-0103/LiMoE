# Getting Started

Before training, you may encounter an error `MMCV=={xxx} is used but incompatible. Please install mmcv>={xxx}, <{xxx}`. We suggest to modify `__init__.py` under `mmdet` and `mmdet3d` package as follows:

```python
mmcv_maximum_version = '3.0.0'
```

Meanwhile, you should modify `Line 123-124` in `mmdet3d/datasets/seg3d_dataset.py` as follows:

```python
if scene_idxs is not None:
    self.scene_idxs = self.get_scene_idxs(scene_idxs)
    self.data_list = [self.data_list[i] for i in self.scene_idxs]
```

## Train with a single GPU

```bash
python train.py ${CONFIG_FILE}
```

## Train with multiple GPUs

```bash
bash dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

**Note**: For pretraining phase, we suggest to use 8 GPUs while 4 GPUs for downstream tasks.
