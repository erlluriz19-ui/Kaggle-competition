# Facial Feature Classification Challenge (ENGE707 Assessment 3)

Individual Kaggle submission by Matty Luriz (student ID 23206450, Kaggle username **Outer Skye**).

The task is to classify face crops into 8 classes (CatEye, CatFace, CatMouth, CatNose, DogEye, DogFace, DogMouth, DogNose). Submissions are scored by accuracy.

## Approach

A ResNet-18 pretrained on ImageNet is fine-tuned on 96 by 96 images using PyTorch, on a CPU only laptop. Three training runs are compared:

| Run | Layers trained | Learning rate | Augmentation | Validation accuracy | Kaggle public score |
|---|---|---|---|---|---|
| Baseline | layer4, fc | 3e-4, constant | yes | 0.9878 | 0.9662 |
| Improvement run (final) | layer3, layer4, fc | 3e-4, cosine | yes | 0.9947 | 0.9723 |
| Ablation | layer3, layer4, fc | 3e-4, cosine | no | 0.9954 | 0.9668 |

The final submission is `submission_task4.csv` (the improvement run).

## Files

| File | Content |
|---|---|
| `Kaggle_Competition_23206450.ipynb` | All code and written analysis, Tasks 1 to 7 |
| `requirements.txt` | Python packages |
| `submission_task4.csv` | Final Kaggle submission (improvement run) |
| `submission.csv` | Baseline submission |
| `submission_task6_noaug.csv` | Ablation submission (no augmentation) |
| `best_model.pt`, `best_model_task4.pt`, `best_model_task6_noaug.pt` | Saved weights of the best validation epoch of each run |
| `history.csv`, `history_task4.csv`, `history_task6_noaug.csv` | Training and validation curves of each run |
| `split.csv` | The saved train/validation split |

## How to run

1. Install Python 3.10 or newer and the packages: `pip install -r requirements.txt`. For a CPU only machine, use the CPU build of PyTorch.
2. Open the notebook and run all cells from top to bottom. Each task builds on the earlier ones, so run Task 1 first.
3. A training cell is skipped when its weights and history files already exist, so the analysis in Tasks 5 and 6 can be rerun in a few minutes using the saved files.
4. To test the whole pipeline quickly, set `DEBUG_N` to a small number (for example 512) in the Task 3 cell and set it back to `None` for the real run.

## Reproducibility notes

- A fixed seed (`SEED = 42`) controls the split, the data order and the weight initialisation. The split is grouped by file hash so that exact duplicate images stay on one side.
- Run time on an Intel Core i7-10750H (CPU only): about 13 to 14 minutes per epoch, 5 epochs per run, three runs, about 3.5 hours in total.
- Exact numbers can differ slightly on other hardware or library versions.
- `NUM_WORKERS` is 0 on Windows to avoid multiprocessing problems. It changes speed, not accuracy.

## Report

The technical report (Tasks 1 to 7, with the reference list and the declaration on the use of generative AI) is submitted separately.
