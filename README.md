# Segmentation experiment runner

Standalone training/evaluation repository for patient-wise UNet, ResNet34-U-Net, SegFormer, and SAM2.1 Hiera-S small experiments.

All commands in this guide assume that the terminal is opened in this `experiments/` directory:

```bash
cd experiments
```

After changing into this directory, do not prefix the paths below with `experiments/`.

## Architecture names

Pass exactly one of these strings to `--architecture`:

| Model | Expected string | Runner |
|---|---|---|
| Classical U-Net | `unet` | `run_experiment.py` |
| ResNet34-encoder U-Net | `resnet34_unet` | `run_experiment.py` |
| SegFormer B0 | `segformer_b0` | `run_experiment.py` |
| SegFormer B1 | `segformer_b1` | `run_experiment.py` |
| SegFormer B2 | `segformer_b2` | `run_experiment.py` |
| SAM2.1 Hiera-S | `sam2` | `sam2_train.py` |

The `sam2` architecture uses its dedicated CUDA pipeline and is not accepted by `evaluate.py`.

## Local setup

```bash
bash setup_local_env.sh
source .venv/bin/activate
python check_environment.py --architecture unet
python validate_setup.py --dataset-root dataset --test-patient p5
python -u run_experiment.py --config configs.json --architecture unet --dry-run
python -u run_experiment.py --config configs.json --architecture resnet34_unet --dry-run
```

## Dry runs

A dry run validates the dataset and prints the patient-wise cross-validation plan without training. Use the exact architecture strings listed above:

```bash
python -u run_experiment.py --config configs.json --architecture unet --dry-run
python -u run_experiment.py --config configs.json --architecture resnet34_unet --dry-run
python -u run_experiment.py --config configs.json --architecture segformer_b0 --dry-run
python -u run_experiment.py --config configs.json --architecture segformer_b1 --dry-run
python -u run_experiment.py --config configs.json --architecture segformer_b2 --dry-run
```

The default protocol holds out `p5` as the locked test patient and performs leave-one-patient-out cross-validation over the remaining patients.

## Training commands

Add `--execute` to start training. The same command pattern works for the classical U-Net, ResNet34-U-Net, and SegFormer variants:

```bash
python -u run_experiment.py --config configs.json --architecture unet --execute
python -u run_experiment.py --config configs.json --architecture resnet34_unet --execute
python -u run_experiment.py --config configs.json --architecture segformer_b0 --execute
python -u run_experiment.py --config configs.json --architecture segformer_b1 --execute
python -u run_experiment.py --config configs.json --architecture segformer_b2 --execute
```

Use `--run-name` to choose a readable output directory:

```bash
python -u run_experiment.py \
  --config configs.json \
  --architecture resnet34_unet \
  --run-name resnet34_unet_seed42 \
  --execute
```

Useful options are `--dataset-root PATH`, `--output-dir PATH`, `--run-name NAME`, `--overlay-count N`, and `--local-files-only` for offline Hugging Face model use. The default dataset root is `dataset/processed`.

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

## Evaluating a saved checkpoint

`evaluate.py` evaluates a checkpoint on the locked test patient from `configs.json`:

```bash
python -u evaluate.py \
  --config configs.json \
  --architecture resnet34_unet \
  --checkpoint runs/experiments/resnet34_unet_seed42/fold_00/best_model.pt \
  --output runs/experiments/resnet34_unet_seed42/test_metrics.json
```

Replace `resnet34_unet` with `unet`, `segformer_b0`, `segformer_b1`, or `segformer_b2` as required. Add `--dry-run` to validate the selection without loading the checkpoint.

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
pip install kaggle

export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx

kaggle datasets download rngarcia/acalasia-partial-0826 
unzip -q /workspace/acalasia-CV-experiment/acalasia-partial-0826.zip 

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

For the ResNet34-U-Net baseline, use `--architecture resnet34_unet`. The model uses the shared runner's patient-wise folds, patient-balanced training sampler, preprocessing, loss, metrics, checkpointing, history, and overlay outputs. Set `architectures.resnet34_unet.pretrained` to `false` for a randomly initialized encoder. ImageNet weights may be downloaded by torchvision on the first pretrained run.
