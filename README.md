# VisionAct

**Human Activity Recognition and Motion Analysis Using Computer Vision**

VisionAct is an academic project for *Foundations of Computer Vision* (FOCV). It takes a video, samples frames
with OpenCV, extracts features with a pretrained CNN, and classifies the human activity. A Streamlit app
demonstrates the classical course topics (filtering, thresholding, morphology, edges, shape analysis, optical
flow) and shows each model's predictions and evaluation.

Three ways of recognising an activity are trained and tested on exactly the same videos:

```
                 /-> MobileNetV3-Small (frozen, ImageNet) -> mean over time -> linear SVM   baseline
Input video ----+--> MobileNetV3-Small (frozen, ImageNet) -> LSTM over the ordered frames
                 \-> R3D-18 (frozen, Kinetics-400) on 5 clips -> new linear head -> mean over clips
```

Everything runs on a CPU. No GPU is assumed anywhere.

## Results at a glance

Measured on 125 held-out test videos (28 source videos) from four UCF101 classes: BodyWeightSquats, JumpingJack,
PullUps and PushUps. All models use the same train/validation/test split, and each was selected on validation
data only.

| Model | Test accuracy | 95% interval | Macro F1 | Validation accuracy | Inference per video |
|---|---|---|---|---|---|
| CNN + mean pooling + linear SVM (baseline) | 77.6% (97/125) | 63.0% to 91.2% | 0.775 | 92.0% | 0.16 s |
| CNN + LSTM | 74.4% (93/125) | 59.0% to 89.7% | 0.745 | 88.0% | 0.14 s |
| CNN + SVM, augmented training | 75.2% (94/125) | 59.5% to 89.6% | 0.755 | 92.0% | 0.13 s |
| CNN + LSTM, augmented training | 77.6% (97/125) | 62.2% to 92.7% | 0.777 | 86.7% | 0.13 s |
| **R3D-18 (Kinetics-400, frozen) + linear head** | **100.0% (125/125)** | see note | 1.000 | 98.7% | 2.0 s |

The intervals resample whole source videos, because clips cut from one source video tend to be right or wrong
together.

- **Largest measured improvement: the pretrained video model.** Replacing ImageNet frame features with frozen
  R3D-18 clip features raised test accuracy from 77.6% to 100.0% (28 more test videos correct, none lost).
  The same gain appears without touching the test set: over 200 random hold-outs of the training and validation
  videos, mean accuracy rises from 78.7% to 98.9%.
- **Read the 100% with care.** It is 125 clips from only 28 source videos; with no errors, all that can be said
  at the 95% level is that accuracy is above roughly 97% if clips were independent, or that at least about 88% of
  source videos would be classified perfectly. R3D-18 was also pretrained on Kinetics-400, which already
  contains push-up, pull-up and squat classes, so the result reflects prior knowledge of these activities.
- **Augmentation and the LSTM made no reliable difference.** The four MobileNet-based models are within three
  videos of each other and their intervals overlap almost completely.
- **Optical-flow fusion was not adopted.** Adding flow histograms did not improve validation accuracy with
  either feature set.
- **Cost.** R3D-18 takes a few seconds per video on the development laptop, against a fraction of a second for
  the MobileNet models, and its backbone is about 130 MB instead of 6 MB.

These numbers come from `results/comparison.json`, which the pipeline writes. The diagnosis of the original
accuracy and the full analysis are in [docs/IMPROVEMENT_REPORT.md](docs/IMPROVEMENT_REPORT.md); the complete
write-up is [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md).

## Requirements

- Python 3.10 or newer (developed and tested on Python 3.14, Windows 11).
- About 1.5 GB of disk for the Python environment, 122 MB for the dataset subset, about 100 MB for cached
  features and trained models.
- Internet access once, for the dataset (122 MB) and the pretrained weights: MobileNetV3-Small (10 MB) and
  R3D-18 (127 MB).

Tested package versions: torch 2.14.1 (CPU), torchvision 0.29.1, opencv-python-headless 5.0.0.93, numpy 2.5.4,
scikit-learn 1.9.1, scipy 1.18.1, streamlit 1.65.0, matplotlib 3.11.2. `requirements-lock.txt` lists every
installed package with its exact version.

## Installation and running the app

### Windows desktop-style launch

1. Install Git and Python 3.10 or newer.
2. Clone the repository:

```powershell
git clone https://github.com/layobaiju/focv-project.git
cd focv-project
```

3. Double-click **Start VisionAct.bat**.

On first launch, the launcher creates an isolated Python environment under `%LOCALAPPDATA%\VisionAct`, installs the dependencies, and starts the app in your browser at <http://localhost:8501>. Initial setup needs an internet connection and can take several minutes. Later launches reuse the installed environment. Keep the launcher window open while using the app; press **Ctrl+C** in that window to stop it. If `requirements.txt` changes, the launcher installs the updated dependencies next time it runs.

You can also launch it from PowerShell in the project folder:

```powershell
.\Start VisionAct.bat
```

### Manual setup

These commands are for Windows PowerShell and create the environment outside the project folder, which avoids syncing its many files when the project is in OneDrive:

```powershell
py -m venv $HOME\.venvs\visionact
& $HOME\.venvs\visionact\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks environment activation, run the app with the environment's Python directly:

```powershell
& $HOME\.venvs\visionact\Scripts\python.exe -m pip install -r requirements.txt
& $HOME\.venvs\visionact\Scripts\python.exe -m streamlit run app.py
```

Python 3.10 or newer is required. On macOS or Linux, use `python3 -m venv ~/.venvs/visionact`, activate it with `source ~/.venvs/visionact/bin/activate`, install with `python -m pip install -r requirements.txt`, and launch with `python -m streamlit run app.py`. On Linux, install the CPU build of PyTorch first:

```sh
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The repository does not include the UCF101 videos, generated feature caches, or trained model files. Video Analysis, Image Processing, and Motion Analysis work without them. Model recognition and evaluation features need data and trained models; the app explains what is missing.

### Optional: download data and train models

Training is optional if you only want to explore the video, image, and motion tools. From the project folder with the environment activated, download the data subset and train/evaluate the models:

```powershell
python -m visionact download
python -m visionact all
```

The download command asks before fetching 437 clips (about 122 MB). The full pipeline validates and splits the data, extracts features, trains all models, and evaluates them. It asks before downloading the pretrained R3D-18 weights (about 127 MB); add `--yes` to skip that prompt. The first run can take about 45 minutes on a laptop CPU. Later runs reuse cached features.

To classify a video from the command line:

```powershell
python -m visionact predict path\to\video.mp4
```

This selects the model with the best validation score. Add `--model cnn_lstm`, or another model key from the Models table, to select a model explicitly.

## Dataset

### What is used

A four-class subset of **UCF101** (Soomro, Zamir and Shah, 2012): BodyWeightSquats, JumpingJack, PullUps and
PushUps. These are real UCF101 class names. UCF101 has no generic classes such as *sitting*, *standing* or
*falling*.

| Class | Clips | Train | Validation | Test |
|---|---|---|---|---|
| BodyWeightSquats | 112 | 61 | 21 | 30 |
| JumpingJack | 123 | 64 | 22 | 37 |
| PullUps | 100 | 56 | 16 | 28 |
| PushUps | 102 | 56 | 16 | 30 |
| **Total** | **437** | **237** | **75** | **125** |

Clips are 320 x 240 at 25 or 29.97 fps, between 29 and 267 frames long (107 on average). They were cut from
100 source videos (25 per class): 56 in training, 16 in validation and 28 in test.

### Where it comes from

The official archive at <https://www.crcv.ucf.edu/data/UCF101.php> is a single 6.9 GB RAR file. To avoid that
download, `python -m visionact download` fetches only the configured classes from the Hugging Face dataset
[`sumith2425/UCF101`](https://huggingface.co/datasets/sumith2425/UCF101), a third-party copy that stores each
clip as a separate file. Each file is verified against the size and SHA-256 published by the host, and only
files named like UCF101 clips are accepted. The copy is not the official distribution, so its content has not
been compared byte-for-byte with the official archive.

To use the official archive instead, extract it and set `dataset_dir` to the extracted `UCF-101` folder. The
layout is the same:

```
data/UCF101/
    BodyWeightSquats/v_BodyWeightSquats_g01_c01.avi
    JumpingJack/v_JumpingJack_g01_c01.avi
    ...
```

UCF101 videos were collected from YouTube for research use. Check the dataset's terms before redistributing clips.

### How it is split

In UCF101, clips named `v_<Class>_gXX_cYY` with the same group `gXX` were cut from the same source video. They
share the actor, background and camera angle, so the dataset authors require that a group never appears in both
training and testing. VisionAct splits by group, never by clip or frame:

- **Test:** groups 01 to 07, the test groups of the official split 1.
- **Validation:** 4 of the remaining 18 groups, drawn with the configured seed (groups 9, 15, 19, 20 for seed 42).
- **Training:** the other 14 groups.

`prepare` writes the split to `artifacts/splits/manifest.csv` and its description (seed, groups, counts,
excluded files) to `artifacts/splits/split_config.json`. The loader re-checks on every run that no group
appears in two splits, and every model's `training.json` records the SHA-256 of the manifest it was trained on.

### Checking the data

```powershell
python -m visionact diagnose
```

`diagnose` audits the data and the trained models and writes `results/diagnostics.json`. On this dataset it found:

- all 437 files decode, and no file name disagrees with its class folder;
- no byte-identical files, and no near-duplicate videos shared between splits;
- none of the 100 source videos appears in more than one split;
- class imbalance is negligible (largest to smallest training class: 1.14);
- every saved model reproduces its stored test predictions exactly.

It also measures how reliable a validation score is (by repeating the hold-out with other validation groups)
and how accuracy grows with the number of training source videos. See the improvement report for the findings.

### Using your own videos

Put the videos in one folder per class and list the folder names in `classes`:

```
my_dataset/
    waving/alice__clip1.mp4
    waving/alice__clip2.mp4
    waving/bob__clip1.mp4
    clapping/...
```

Set `"split_mode": "random_group"` in `config.json`. Clips that come from the same recording or show the same
person should share a prefix before a double underscore (`alice__`); that prefix is the group and is kept in
one split. Files without `__` are each their own group. Every class needs at least three groups. If mirroring a
video would change its label (for example "turning left"), also set `"augment_hflip": false`.

## Models

| Key | Model | What is trained |
|---|---|---|
| `cnn_svm` | MobileNetV3-Small frame features, averaged over 16 frames, linear SVM | the SVM (baseline) |
| `cnn_lstm` | MobileNetV3-Small frame features read in order by an LSTM | the LSTM and its output layer |
| `cnn_svm_aug` | as `cnn_svm`, with augmented training videos added | the SVM |
| `cnn_lstm_aug` | as `cnn_lstm`, with augmented training videos added | the LSTM and its output layer |
| `r3d18_head` | R3D-18 pretrained on Kinetics-400, 5 clips of 16 frames at about 15 fps | a new linear head |

Both backbones are frozen: their features are computed once and cached, so training takes seconds to minutes
on a CPU. The two original models (`cnn_svm`, `cnn_lstm`) are kept unchanged as the reference; the augmented
variants are saved separately and never overwrite them.

**Preprocessing.** Each backbone gets the input size and normalisation its pretrained weights were built for,
read from the weights themselves: 224 x 224 crops with ImageNet statistics for MobileNetV3-Small, 112 x 112
crops with Kinetics statistics for R3D-18. Tests check the OpenCV preprocessing against torchvision's own
transforms.

**Augmentation (training videos only).** Four augmented copies of each training video are cached: a randomly
positioned crop instead of the centre crop, jittered frame or clip positions, brightness scaled by a factor
between 0.8 and 1.2, and a mirror image half of the time. Mirroring is valid here because all four exercises
are left-right symmetric. Validation and test videos are never augmented. Whether the R3D-18 head uses the
augmented clips is decided on validation.

**Selection.** Hyperparameters are chosen on the validation split; neural models stop early on validation loss
and keep the best checkpoint. The default model for predictions is the one with the best validation macro-F1.
The test split is scored only by `evaluate`.

**Not adopted.** Fine-tuning the last R3D-18 block was not run: the frozen model already gets 74 of 75
validation videos right, which leaves no room to measure an improvement. Optical-flow fusion was tested and
not adopted (see the improvement report).

## Pipeline commands

Run in this order. Each command accepts `--help`.

| Command | What it does | Output |
|---|---|---|
| `python -m visionact download` | Downloads the configured UCF101 classes after confirmation | `data/UCF101/` |
| `python -m visionact prepare` | Checks labels and paths, decodes a frame of every video, splits by group | `artifacts/splits/` |
| `python -m visionact extract` | Caches MobileNetV3-Small features (16 frames per video) | `artifacts/features/` |
| `python -m visionact train-baseline` | Fits the SVM, choosing C on validation | `artifacts/models/cnn_svm_baseline.joblib` |
| `python -m visionact train-lstm --search` | Trains the LSTM with early stopping; `--search` tries the validation grid | `artifacts/models/cnn_lstm_best.pt` |
| `python -m visionact extract-augmented` | Caches augmented MobileNetV3-Small features for the training split | `artifacts/features/` |
| `python -m visionact train-baseline --augment` | SVM trained on original and augmented training videos | `artifacts/models/cnn_svm_aug.joblib` |
| `python -m visionact train-lstm --search --augment` | LSTM trained on original and augmented training videos | `artifacts/models/cnn_lstm_aug.pt` |
| `python -m visionact extract-r3d` | Caches R3D-18 clip features (asks before the 127 MB weight download) | `artifacts/features/` |
| `python -m visionact train-r3d` | Trains the R3D-18 classification head, selected on validation | `artifacts/models/r3d18_head_best.pt` |
| `python -m visionact evaluate` | Tests every trained model on the held-out test split | `results/` |
| `python -m visionact all` | All of the above after the download | everything |
| `python -m visionact diagnose` | Audits data, split and trained models | `results/diagnostics.json` |
| `python -m visionact motion-experiment` | Optional: tests whether optical-flow features add to the visual features | `results/motion_experiment.json` |
| `python -m visionact predict VIDEO` | Classifies one video (`--model KEY` to choose the model) | printed scores |

`evaluate` is the only command that scores the classifiers on the test split. Run it after model selection is
finished.

## Configuration

Defaults live in [visionact/config.py](visionact/config.py). To change them, copy `config.example.json` to
`config.json` and edit it, or pass flags for the common settings:

```powershell
python -m visionact train-lstm --hidden-dim 64 --epochs 40 --batch-size 16 --learning-rate 0.0005 --num-frames 16
```

| Setting | Default | Meaning |
|---|---|---|
| `classes` | the four exercise classes | Class folder names to use |
| `dataset_dir` | `data/UCF101` | Dataset location (also `VISIONACT_DATASET_DIR`) |
| `seed` | 42 | Seed for the split, augmentation, shuffling and weight initialisation |
| `num_frames` | 16 | Frames sampled per video for the MobileNet models (the LSTM sequence length) |
| `hidden_dim`, `num_layers`, `dropout` | 128, 1, 0.3 | LSTM architecture |
| `epochs`, `batch_size`, `learning_rate`, `weight_decay` | 60, 16, 0.001, 0.0001 | LSTM training |
| `patience` | 10 | LSTM early-stopping patience in epochs (validation loss) |
| `search_*` | see file | Grid tried by `train-lstm --search` |
| `svm_c_grid` | 0.0001 to 10 | SVM regularisation values tried on validation |
| `augment_copies` | 4 | Augmented copies per training video; 0 disables augmentation |
| `augment_brightness`, `augment_hflip` | 0.2, true | Brightness range (plus or minus) and mirroring |
| `r3d_clip_len`, `r3d_target_fps`, `r3d_clips_per_video` | 16, 15, 5 | R3D-18 input: the settings its weights were evaluated with |
| `r3d_head_*` | see file | Head training: learning rate, batch size, epochs, patience, validation grid |

Changing a sampling or augmentation setting changes the cache key, so the affected features are extracted
again into a new folder and the models that use them must be retrained.

## The app

| Section | Content | Needs a trained model |
|---|---|---|
| Home | Purpose, architecture diagram, pipeline status | no |
| Video Analysis | Metadata, sampled frames, resizing and normalisation | no |
| Image Processing | Gaussian and median filtering, thresholding, morphology, Canny edges, contours and connected components | no |
| Motion Analysis | Farneback dense flow, Lucas-Kanade sparse flow, motion over time | no |
| Activity Recognition | Prediction and class scores from any trained model; all models side by side | yes |
| Model Comparison | Every model on the same test videos, significance, cost, how each was selected | yes |
| Evaluation | Per-class metrics, confusion matrix, misclassified videos, training curves | yes |
| About | Course topics, which parts feed the classifier, limitations | no |

Choose the input video in the sidebar: upload a file (AVI, MP4, MOV, MKV and similar) or, once the dataset is
prepared, pick a dataset clip. Clips are labelled with their split; use a clip marked `(test)` for a fair check
of a model. Browsers cannot play AVI files, so for those the app shows decoded frames instead of a player.

The first three sections are demonstrations and say so. The model sections only ever show numbers read from a
trained checkpoint or from `results/`; without them they show the commands to run.

### Which techniques feed the classifiers

| Technique | Role |
|---|---|
| Frame and clip sampling, resizing, cropping, normalisation | Part of every classifier pipeline |
| Random crops, brightness changes, mirroring | Training-time augmentation only |
| MobileNetV3-Small, LSTM, R3D-18 | The classifiers |
| Gaussian and median filtering, thresholding, morphology | Independent demonstrations on single frames |
| Canny edges, contours, connected components | Independent demonstrations on single frames |
| Optical flow | Demonstration, plus a separate experiment; not in the prediction pipeline |

The classical operators are not fed to the classifiers on purpose: the networks were pretrained on unfiltered
images, and a threshold mask of a natural video frame isolates bright regions rather than people.

## Tests

```powershell
python -m pytest
```

174 tests cover frame and clip sampling, preprocessing, augmentation, the classical operators, optical flow,
dataset validation and group-level splitting, feature extraction and caching for both backbones, all models,
the metrics, the diagnostics, the download logic, the full pipeline on a small synthetic dataset, and every app
section. Test videos are generated on the fly, so the dataset is not needed. Tests marked `slow` use the
pretrained weights; to skip them:

```powershell
python -m pytest -m "not slow"
```

## Project layout

```
app.py                    Streamlit entry point
visionact/
    config.py             all paths and hyperparameters
    video.py              video loading, frame and clip sampling, CNN preprocessing
    augment.py            training-only augmentation
    image_processing.py   filtering, thresholding, morphology, edges, shape analysis
    optical_flow.py       dense and sparse optical flow, flow histograms
    dataset.py            scanning, validation, group-level splitting, experiment records
    download.py           per-class UCF101 download with checksum verification
    features.py           MobileNetV3-Small feature extraction and caching
    baseline.py           mean pooling + linear SVM
    lstm.py               LSTM classifier, training, checkpointing
    r3d.py                R3D-18 feature extraction, classification head
    training.py           training loop with early stopping (R3D-18 head)
    registry.py           the list of models and where their checkpoints live
    inference.py          single-video prediction with any trained model
    evaluation.py         metrics, comparison, timing, result export
    diagnostics.py        data audit, leakage checks, validation reliability
    motion_experiment.py  optional optical-flow fusion experiment
    plots.py              confusion matrix, training curves, score charts
    cli.py                command-line interface
    ui/                   one module per app section
tests/                    unit, pipeline and app tests
docs/
    IMPROVEMENT_REPORT.md diagnosis, changes, largest measured improvement, limitations
    PROJECT_REPORT.md     academic write-up
results/                  metrics, predictions, figures and experiment records from the last run
data/, artifacts/         dataset, feature caches, trained models (not under version control)
```

## Reproducibility

- Seeds are fixed for the split, augmentation, data shuffling and weight initialisation; PyTorch runs in
  deterministic mode.
- Re-running `python -m visionact all` on the development machine reproduced the per-video test predictions of
  all five models exactly. Timings vary from run to run and between machines.
- `results/<model>/training.json` records, for every model, the validation result of each candidate, the
  selected hyperparameters, the preprocessing, the augmentation and the fingerprint of the split.
  `results/<model>/predictions.csv` has one row per test video, so every reported metric can be recomputed.
- Trained checkpoints are in `artifacts/models/`.
- Results on other hardware, operating systems or library versions may differ slightly because of
  floating-point differences.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Dataset folder not found` | Run `python -m visionact download`, or set `dataset_dir` |
| `No dataset split found` | Run `python -m visionact prepare` |
| `Cached features are missing` | Run the command named in the message (`extract`, `extract-augmented` or `extract-r3d`) |
| `No trained ... checkpoint` | Run the training command named in the message |
| `The pretrained R3D-18 weights are not on this computer` | Re-run with `--yes` to allow the 127 MB download |
| Weights fail to download | The first extraction needs internet access to `download.pytorch.org` |
| The video player is missing in the app | The file is an AVI; the frames are still decoded and shown |
| The first command is very slow | The first import of PyTorch after installing or rebooting is slow; later runs start quickly |

## Acknowledgements

- UCF101: K. Soomro, A. R. Zamir and M. Shah, "UCF101: A Dataset of 101 Human Actions Classes From Videos in
  The Wild", CRCV-TR-12-01, 2012.
- MobileNetV3: A. Howard et al., "Searching for MobileNetV3", ICCV 2019.
- R3D-18: D. Tran et al., "A Closer Look at Spatiotemporal Convolutions for Action Recognition", CVPR 2018.
- Kinetics-400: W. Kay et al., "The Kinetics Human Action Video Dataset", 2017.
- Pretrained weights from torchvision.
