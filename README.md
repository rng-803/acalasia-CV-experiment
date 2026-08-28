# Segmentation experiment runner

Standalone training/evaluation repository for patient-wise UNet, SegFormer, and SAM2.1 Hiera-S small experiments.

## Local setup

```bash
bash setup_local_env.sh
source .venv/bin/activate
python check_environment.py --architecture unet
python validate_setup.py --dataset-root /path/to/dataset/processed --test-patient p5
python -u run_experiment.py --config configs.json --architecture unet --dry-run
```

For SegFormer, either allow Hugging Face network access and omit `--local-files-only`, or pre-cache the selected model before using offline mode:

```bash
python download_hf_model.py --model nvidia/mit-b0
python -u run_experiment.py --config configs.json --architecture segformer_b0 --local-files-only --execute
```

Repeat the download with `nvidia/mit-b1` or `nvidia/mit-b2` for the larger variants. The model cache is normally under `~/.cache/huggingface`; preserve or recreate it on the cloud volume if pods are ephemeral.

## Run outputs and comparison

Every execution creates a unique directory under `runs/experiments/` unless `--run-name` is supplied. A run contains its complete `run_config.json`, per-epoch `history.csv`, one directory per CV fold, `fold_metrics.csv`, `aggregate_metrics.csv`, checkpoints, and `overlays/` panels showing original, ground truth, prediction, and blend. This prevents separate experiments from overwriting one another.

```bash
python -u run_experiment.py --config configs.json --architecture segformer_b0 --run-name segformer_b0_seed42 --execute
python compare_runs.py --runs-root runs/experiments --output runs/experiment_comparison.csv
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

Add `--execute` only after the dry-run succeeds. The SAM2 pipeline freezes the image encoder and samples random positive points from the 5x5-eroded foreground mask. Use `python -u` for live RunPod logs; the runners also flush progress messages explicitly.

## Cloud data layout

dataset is at Kaggle. Download with: 

```
import kagglehub

export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx

# Download latest version
path = kagglehub.dataset_download("rngarcia/acalasia-partial-0826")

print("Path to dataset files:", path)
```

The runner expects, inside the main directory:

```text
dataset/processed/p1/p1_images/
dataset/processed/p1/p1_masks/
...
dataset/processed/p5/p5_images/
dataset/processed/p5/p5_masks/
```

The default protocol holds out p5, performs leave-one-patient-out cross-validation over p1–p4, and balances training sampling by patient.
