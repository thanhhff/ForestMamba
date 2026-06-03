# ForestMamba

ForestMamba is a forest tree instance segmentation model built on a **sparse Mamba (SSM) backbone** with **CHM-guided multi-scale query initialisation**.

- 📄 [Paper on arXiv](https://arxiv.org/abs/2606.01549)
- 📦 [Dataset & pre-trained model on Zenodo](https://zenodo.org/records/16742708)
- 🔗 [Pre-trained ForestMamba checkpoint (Google Drive)](https://drive.google.com/drive/folders/1wf_IcXNIZgxOg_Dkrt45ph2ssdBeWCHm?usp=sharing)

---

## Directory structure

Download the dataset and pre-trained checkpoint from Zenodo or Google Drive and place them as follows:

```
ForestMamba/
├── configs/ForAINetv2/
│   └── forestmamba_chm_radius16_qp300_2many_v6_expand_1.py   ← inference config
├── data/
│   └── ForAINetV2/
│       ├── train_val_data/
│       └── test_data/
├── work_dirs/
│   └── forestmamba/
│       └── epoch_3000_fix.pth                                 ← pre-trained checkpoint
```

---

## Environment setup

### 1. Build Docker image

```bash
cd /path/to/ForestMamba

sudo docker build -t forestmamba-image .

sudo docker run --gpus all --shm-size=128g -d -p 127.0.0.1:49211:22 \
  -v /path/to/ForestMamba:/workspace \
  -v /workspace/segmentator:/workspace/segmentator \
  --name forestmamba-container forestmamba-image

sudo docker exec -it forestmamba-container /bin/bash
```

### 2. Install Mamba-specific packages

ForestMamba requires the following packages:

```bash
pip install mamba-ssm==1.1.1
pip install causal-conv1d==1.1.1
pip install hilbertcurve==2.0.5
pip install transformers==4.36.0
pip install importlib-metadata==4.13.0
pip install wandb
```

### 3. Resolve common import errors

```bash
# Test torch-points-kernels
python -c "from torch_points_kernels import instance_iou; print('torch-points-kernels loaded successfully')"

# If you see: ModuleNotFoundError: No module named 'torch_points_kernels.points_cuda'
pip uninstall torch-points-kernels -y
pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0

# Reinstall torch-cluster
pip uninstall torch-cluster -y
pip install torch-cluster --no-cache-dir --no-deps
```

### 4. Replace required mmengine/mmdet3d files

```bash
pip show mmengine   # check the install path

cp replace_mmdetection_files/loops.py \
   /opt/conda/lib/python3.10/site-packages/mmengine/runner/
cp replace_mmdetection_files/base_model.py \
   /opt/conda/lib/python3.10/site-packages/mmengine/model/base_model/
cp replace_mmdetection_files/transforms_3d.py \
   /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/transforms/
cp replace_mmdetection_files/distributed.py \
   /opt/conda/lib/python3.10/site-packages/mmengine/model/wrappers/distributed.py
```

---

## Data preparation

> **Before running anything**, update `data_root_forainetv2` in the config to point to your data directory:
>
> ```python
> # configs/ForAINetv2/forestmamba_chm_radius16_qp300_2many_v6_expand_1.py
> data_root_forainetv2 = '/your/path/to/ForAINetV2/'
> ```

```bash
export PYTHONPATH=/workspace/ForestMamba

# Place .ply files:
#   training / validation → /workspace/data/ForAINetV2/train_val_data/
#   test                  → /workspace/data/ForAINetV2/test_data/

pip install laspy "laspy[lazrs]"

# Preprocess raw point clouds
cd /workspace/data/ForAINetV2
python batch_load_ForAINetV2_data.py --num_workers 8
# → produces /workspace/data/ForAINetV2/forainetv2_instance_data/

# Create .pkl info files
cd /workspace
python tools/create_data_forainetv2.py forainetv2
```

---

## Training

**Multi-GPU (recommended):**

```bash
export PYTHONPATH=/workspace/ForestMamba
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1 PORT=29500 bash tools/dist_train.sh \
  configs/ForAINetv2/forestmamba_chm_radius16_qp300_2many_v6_expand_1.py \
  2 \
  --work-dir work_dirs/forestmamba
```

**Single GPU:**

```bash
export PYTHONPATH=/workspace/ForestMamba

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/ForAINetv2/forestmamba_chm_radius16_qp300_2many_v6_expand_1.py \
  --work-dir work_dirs/forestmamba
```

**Resume from a checkpoint:**

```bash
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 bash tools/dist_train.sh \
  configs/ForAINetv2/forestmamba_chm_radius16_qp300_2many_v6_expand_1.py \
  2 \
  --work-dir work_dirs/forestmamba \
  --resume work_dirs/forestmamba/epoch_1000.pth
```

---

## Inference

> The pre-trained checkpoint from Google Drive / Zenodo is already fixed and ready to use directly.
>
> If you trained your own model, fix the checkpoint first:
> ```bash
> python tools/fix_spconv_checkpoint.py \
>   --in-path  work_dirs/forestmamba/epoch_3000.pth \
>   --out-path work_dirs/forestmamba/epoch_3000_fix.pth
> ```

### Run inference

To disable Weights & Biases logging during inference:

```bash
export WANDB_MODE=disabled
```

**Single GPU:**

```bash
export PYTHONPATH=/workspace/ForestMamba

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/ForAINetv2/forestmamba_chm_radius16_qp300_2many_v6_expand_1.py \
  work_dirs/forestmamba/epoch_3000_fix.pth
```

**Multi-GPU:**

```bash
export PYTHONPATH=/workspace/ForestMamba

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  tools/test.py \
  configs/ForAINetv2/forestmamba_chm_radius16_qp300_2many_v6_expand_1.py \
  work_dirs/forestmamba/epoch_3000_fix.pth \
  --launcher pytorch
```

---

## Evaluation

```bash
python tools/eval_predictions.py \
  work_dirs/inference/forestmamba \
  --workers 16
```

---

## Testing on custom data

### 1. Place your test files

```
/workspace/data/ForAINetV2/test_data/
```

### 2. Update the test list

Edit `/workspace/data/ForAINetV2/meta_data/test_list.txt` and append your file names (without extension).

### 3. Re-run preprocessing and inference

```bash
cd /workspace/data/ForAINetV2
python batch_load_ForAINetV2_data.py --num_workers 8
cd /workspace
python tools/create_data_forainetv2.py forainetv2

# Then run inference as above
```

---

## Two-pass inference for dense forests

In very dense plots some trees may be missed in a single pass. A second pass on the remaining unsegmented points improves recall:

```bash
bash /workspace/tools/inference_bluepoint.sh
```

Before running, update `BLUEPOINTS_DIR` in the script to match your output directory, and switch the save mode in `oneformer3d/oneformer3d.py` inside the `predict` function of `ForAINetV2OneFormer3D_XAwarequery`:

```python
# self.save_ply_withscore(...)
self.save_bluepoints(...)
```

And set:

```python
is_test = True
if is_test:
```

---

## Non-.ply input files

If your test data is not in `.ply` format (e.g., `.laz`):

1. Edit `data/ForAINetV2/batch_load_ForAINetV2_data.py` — update the file path in `export_one_scan()`.
2. Edit `data/ForAINetV2/load_forainetv2_data.py` — replace the `read_ply()` call with your loader.
3. If your files have no ground-truth labels, replace the label loading lines with dummy arrays:

```python
semantic_seg = np.ones((points.shape[0],), dtype=np.int64)
treeID = np.zeros((points.shape[0],), dtype=np.int64)
```

**Recommendation:** Converting to `.ply` beforehand is the simplest option.

---

## Memory / OOM tips

- For GPUs with less memory, reduce `radius` in the config.
- For inference OOM, lower `chunk` in the config or reduce `num_points` in `oneformer3d/oneformer3d.py`.

---

## Optional tips

**Tensorboard:**

```bash
tensorboard --logdir=work_dirs/forestmamba/vis_data/ --host=0.0.0.0 --port=6006
```

**SSH debugging in VS Code:**

```bash
apt-get install -y openssh-server
service ssh start
passwd root
echo -e "PermitRootLogin yes\nPasswordAuthentication yes" >> /etc/ssh/sshd_config
service ssh restart
```

Forward port 22 with `-p 127.0.0.1:49211:22` in your `docker run` command.

---

## License

ForestMamba is built on the ForestFormer3D codebase, which is based on OneFormer3D by Danila Rukhovich (https://github.com/filaPro/oneformer3d), licensed under CC BY-NC 4.0. This repository is released under the same license.
