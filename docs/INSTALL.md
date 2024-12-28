# Installation

## Prerequisties

This codebase is tested with `torch==1.12.1`, `mmengine==0.10.4`, `mmcv==2.2.0`, `mmdet==3.3.0`, and `mmdet3d==1.4.0`, with `CUDA 11.3`.

**Step 1.** Create a conda environment and activate it.

```bash
conda create --name limoe python==3.8 -y
conda activate limoe
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

```bash
conda install pytorch torchvision -c pytorch
```

**Step 3.** Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) using [MIM](https://github.com/open-mmlab/mim).

```bash
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmdet3d
```

Optionally, you can also install the above projects from the source, e.g.:

```bash
git clone https://github.com/open-mmlab/mmdetection3d
cd mmdetection3d
pip install -v -e .
```

Meanwhile, you also need to install [`nuScenes-devkit`](https://github.com/nutonomy/nuscenes-devkit).
