# Patient-wise segmentation test-run plan

## 1. Scope and success criteria

This setup is for reproducible smoke tests and model comparison, not a clinical performance claim. The current prepared dataset contains five patients and 229 paired annotated images: p1=101, p2=46, p3=21, p4=31, p5=30. The test patient is selected once before model selection (default `p5`, 30 images) and is not used for fold selection, early stopping, or hyperparameter decisions.

## 2. Data audit and leakage control

1. Run `python3 experiments/validate_setup.py --dataset-root dataset/processed --test-patient p5`.
2. Confirm every image is paired by filename stem with a mask, and inspect the generated JSON manifest.
3. Keep every patient intact: the final test set is all images from the held-out patient; the training pool is the other four patients (199 images by current counts).
4. Use leave-one-patient-out cross-validation (four folds) only inside those four training patients. Each fold validates on one complete patient and trains on the other three. Do not use image-level random splitting.
5. Report patient-level means and standard deviations across folds, plus the untouched test-patient result from one final refit.

## 3. Balancing policy

The training sampler gives each patient equal total probability. An image from patient `p` receives weight `1 / number_of_images(p)`, with replacement during training. This prevents p1's 101 frames from dominating p3's 21 frames. Report both macro-over-patients and pooled pixel metrics; never hide the per-patient scores. Validation and final test loaders remain deterministic and unweighted.

## 4. Experiment matrix

Run a small, comparable first pass at 256x256 for 10 epochs, batch size 4, fixed seed 42, and training-only horizontal flips/photometric augmentation. Use the same folds, sampler, optimizer family, stopping rule, and foreground macro Dice for all non-SAM2 models.

| ID | Model | Default test-run choice | Feasibility |
|---|---|---|---|
| U32 | Existing PyTorch U-Net from `RESNET/teste_vid_01/model_unet.py`, base filters 32 | 10 epochs/fold | Lowest compute; CPU smoke test and CUDA run are practical |
| S0 | Existing SegFormer B0 (`nvidia/mit-b0`) | 10 epochs/fold, pretrained weights | Good baseline; requires Transformers weights locally or network on RunPod |
| S1 | SegFormer B1 (`nvidia/mit-b1`) | 10 epochs/fold, batch 2 if memory requires | Moderate GPU memory/time |
| S2 | SegFormer B2 (`nvidia/mit-b2`) | 5–10 epochs/fold, batch 1–2 | Highest useful SegFormer cost; run after B0/B1 |
| SAM2-S | SAM2.1 Hiera-S (small), prompted with mask-derived points | 600 steps/fold, accumulation 8, frozen image encoder | CUDA-only and opt-in; uses binary foreground targets and prompt-conditioned IoU evaluation |

For model selection, choose using mean foreground macro Dice across the four training-only folds. Then refit the selected architecture on all four training patients and evaluate exactly once on p5. Keep a locked test manifest and save config, seed, fold, checkpoint, and metrics with each run.

## 5. Compute estimate

Estimates assume 199 training images, 256x256, mixed precision on a modern 16 GB CUDA GPU, and 10 epochs per fold. U32 is roughly 2–8 minutes/fold; B0 3–10 minutes/fold; B1 5–15 minutes/fold; B2 8–25 minutes/fold. Four-fold selection plus final refit is therefore roughly 30 minutes to 2.5 hours depending on storage, augmentation, and GPU. CPU execution is for validation/smoke tests only and may take tens of minutes per fold. At 512x512, expect approximately 3–6x the GPU time and lower batch sizes. SAM2 can exceed these estimates substantially and is intentionally opt-in.

## 6. Run order

1. Validate the manifest and dataset audit.
2. Perform one dry-run/config validation for each architecture; verify no training loop starts.
3. Run U32 and S0 across the four folds.
4. Run S1, then S2 only if memory/time is acceptable.
5. Review per-patient fold metrics and select one model.
6. Refit that model on all training patients and evaluate on the locked test patient.
7. Run SAM2-S only in the CUDA cloud environment after installing the official package and downloading its checkpoint. The prompt source is fixed: one random positive point per present non-background label, sampled from the 5x5-eroded foreground union. This follows the point-sampling strategy in the referenced LearnOpenCV article.

## 7. CUDA/RunPod portability

The code uses relative paths, `--dataset-root`, `--output-dir`, `--device auto`, deterministic seeds, optional CUDA AMP, and `num_workers` flags. On RunPod, clone/copy the project, install the requirements, mount the dataset, set `SEGFORMER_DATASET_ROOT` or pass an absolute dataset path, and run the same commands. Do not copy local absolute paths into configs.

## 8. Commands that do not train

```bash
python experiments/validate_setup.py --dataset-root dataset/processed --test-patient p5 --manifest experiment_manifest.json
python3 experiments/run_experiment.py --config experiments/configs.json --architecture unet --dry-run
python experiments/run_experiment.py --config experiments/configs.json --architecture segformer_b0 --dry-run --local-files-only
python experiments/run_experiment.py --config experiments/configs.json --architecture segformer_b1 --dry-run --local-files-only
python experiments/run_experiment.py --config experiments/configs.json --architecture segformer_b2 --dry-run --local-files-only
python3 experiments/evaluate.py --config experiments/configs.json --architecture unet --checkpoint runs/experiments/final_model.pt --dry-run
python3 -m unittest discover -s 'segformer implementation/tests'
```

The `--dry-run` flag only discovers files, builds the split description, checks optional dependencies/model names, and prints the planned workload. No optimizer, checkpoint, or training epoch is started.
