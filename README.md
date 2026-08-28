# Segmentation experiment runner

Standalone training/evaluation repository for patient-wise UNet, SegFormer, and SAM2.1 Hiera-S small experiments.

## Local setup

```bash
bash setup_local_env.sh
source .venv/bin/activate
python check_environment.py --architecture unet
python validate_setup.py --dataset-root /path/to/dataset/processed --test-patient p5
python run_experiment.py --config configs.json --architecture unet --dry-run
```

Place the prepared dataset beside this repository as `dataset/processed/`, or pass `--dataset-root`. It must contain `p1/` through `p5/` patient folders with matching `<patient>_images/` and `<patient>_masks/` directories. Data, checkpoints, and outputs are intentionally not committed.

## CUDA cloud / SAM2 small

```bash
python3 -m pip install -r requirements-sam2.txt
git clone https://github.com/facebookresearch/sam2.git
(cd sam2 && SAM2_BUILD_CUDA=1 python3 -m pip install -e .)
bash download_sam2_small.sh checkpoints
python3 sam2_train.py --dataset-root /path/to/dataset/processed --checkpoint checkpoints/sam2.1_hiera_small.pt --dry-run
```

Add `--execute` only after the dry-run succeeds. The SAM2 pipeline freezes the image encoder and samples random positive points from the 5x5-eroded foreground mask.
