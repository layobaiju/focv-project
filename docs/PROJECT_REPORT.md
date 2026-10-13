# VisionAct: Human Activity Recognition and Motion Analysis Using Computer Vision

Project report for *Foundations of Computer Vision* (FOCV).

All figures in this report are measured values taken from the files in `results/`, produced with seed 42 by
`python -m visionact all`, `python -m visionact diagnose` and `python -m visionact motion-experiment`.
Installation and commands are in the [README](../README.md). A shorter account of what limited the first
version of the project and what improved it is in [IMPROVEMENT_REPORT.md](IMPROVEMENT_REPORT.md).

## 1. Problem statement and objectives

**Problem.** Given a short video of a person, decide which of a small set of activities it shows.

A video differs from an image in having a time axis. A classifier can ignore that axis and look only at what
individual frames contain (appearance), it can model how frame content changes over time, or it can use a
network that processes space and time together. The project compares these three options.

**Objectives.**

1. Build a working application that accepts a video, processes its frames and classifies the activity, on a
   CPU-only computer.
2. Implement the FOCV syllabus topics as working features: image transformations, filtering, thresholding,
   morphology, edge detection, shape analysis, optical flow, CNNs and RNNs.
3. Compare, on identical data splits: an order-blind classifier on CNN frame features (mean pooling and a
   linear SVM), an LSTM that reads the same features in order, and a pretrained 3D CNN (R3D-18) with a new
   classification head.
4. Find out what limits accuracy, and report all results as measured, with enough detail to reproduce them.

**Contribution.** The project is an implementation and an experimental comparison of established techniques.
It does not propose a new algorithm and makes no claim about reliability outside the dataset it was tested on.

## 2. Architecture and data flow

```
Input video
   |
   |-- frame pipeline -----------------------------------------------------------------------------
   |   OpenCV: sample 16 frames evenly over the clip
   |   resize shorter side to 256, centre crop 224 x 224, BGR -> RGB, ImageNet normalisation
   |   MobileNetV3-Small (ImageNet weights, frozen, 1000-way layer removed) -> 16 x 1024 features (cached)
   |        |-- mean over the 16 frames -> standardise -> linear SVM         CNN + SVM   (baseline)
   |        '-- LSTM (128 hidden units) -> last hidden state -> linear       CNN + LSTM
   |
   '-- clip pipeline ------------------------------------------------------------------------------
       OpenCV: 5 clips of 16 frames at about 15 fps, spread over the video
       resize shorter side to 128, centre crop 112 x 112, BGR -> RGB, Kinetics normalisation
       R3D-18 (Kinetics-400 weights, frozen, 400-way layer removed) -> 5 x 512 features (cached)
            '-- new linear head per clip -> mean of the 5 clip logits        R3D-18 + head

All models -> same held-out test videos -> accuracy, precision, recall, F1, confusion matrix, timing
```

| Stage | Module | Notes |
|---|---|---|
| Decoding and sampling | `visionact/video.py` | Frames: the clip is cut into 16 equal segments and the middle frame of each is used. Clips: 16 frames with a stride chosen to approximate 15 fps. Short videos repeat frames; unreadable videos raise a clear error. |
| Preprocessing | `visionact/video.py` | Input size and normalisation constants are read from the pretrained weights. Unit tests check the OpenCV pipeline against torchvision's own transforms for both networks. |
| Augmentation | `visionact/augment.py` | Training videos only: random crop position, temporal jitter, brightness, mirroring. |
| Frame features | `visionact/features.py` | Frozen MobileNetV3-Small, inference mode, CPU, cached per video. |
| Clip features | `visionact/r3d.py` | Frozen R3D-18, cached per video. |
| Baseline | `visionact/baseline.py` | Mean pooling, `StandardScaler`, `LinearSVC` with class-balanced weights. |
| LSTM | `visionact/lstm.py` | Input standardisation with training-set statistics, dropout, one LSTM layer, dropout, linear head. Adam, class-weighted cross-entropy, gradient clipping, early stopping. |
| R3D-18 head | `visionact/r3d.py`, `training.py` | Standardisation, optional dropout, one linear layer. Same optimiser and early stopping. |
| Evaluation | `visionact/evaluation.py` | The only module that scores models on the test split. |
| Diagnostics | `visionact/diagnostics.py` | Data audit, leakage checks, reliability of validation. |
| Interface | `app.py`, `visionact/ui/` | Streamlit, eight sections. |

## 3. Design choices

### 3.1 Why MobileNetV3-Small

The hardware constraint is a laptop CPU with no GPU, so the first CNN had to be cheap to run and could not be
trained from scratch.

- **Small and fast.** The torchvision model has 2.54 million parameters and needs about 0.06 GFLOPs per image.
  With the classification layer removed, the extractor used here has 1.52 million parameters (6.2 MB) and
  processes the 16 frames of a video in roughly 0.1 s on the development laptop.
- **Pretrained.** The ImageNet weights (67.7% top-1 accuracy on ImageNet, as published by torchvision) provide
  general visual features, so only a small classifier has to be trained on the 237 training videos.
- **Frozen.** Fine-tuning the CNN on so few videos would risk overfitting and would require running the CNN on
  every training step. Freezing it allows features to be computed once and cached, after which a training run
  takes seconds.

The final ImageNet layer maps to 1000 object categories that are irrelevant here, so it is removed. The
1024-dimensional activation that fed it is used as the frame descriptor.

### 3.2 Why an LSTM

A sequence of frame features is ordered data. A recurrent network processes it one step at a time and carries a
hidden state forward, so its output can depend on the order of events: in principle it can distinguish "arms go
up, then down" from "arms go down, then up". An LSTM adds gating to a plain RNN, which lets it retain
information over more steps and makes training more stable. With 16 steps and 128 hidden units the model has
591,364 trainable parameters and trains on a CPU in a few seconds.

### 3.3 The baseline and what the comparison isolates

| | CNN + SVM baseline | CNN + LSTM |
|---|---|---|
| Input | the same 16 x 1024 features | the same 16 x 1024 features |
| Use of time | none: the 16 vectors are averaged | frames are read in order |
| Effect of shuffling the frames | none, by construction | can change the prediction |
| Classifier | linear, 4 x 1024 weights | recurrent, 591,364 weights |
| Training | convex optimisation, well under a second | gradient descent with early stopping, 6.7 s |

Because the features, the split and the test videos are identical, any difference between the two can be
attributed to the classifier, and in particular to whether it models temporal order. A unit test
(`tests/test_models.py`) confirms the mechanism on synthetic data: when two classes contain the same frames in
opposite order, the LSTM exceeds 95% validation accuracy while the mean-pooled SVM stays below 70% (chance is
50%).

### 3.4 Why R3D-18 pretrained on Kinetics-400

The first two models share a weakness: their features come from a network trained on still photographs of
objects, which describes scenes and people rather than actions. R3D-18 addresses this directly.

- **It sees motion.** Its 3D convolutions run over 16 consecutive frames at once, so its features encode how
  things move, not only what is visible.
- **It was trained on actions.** Kinetics-400 is a collection of human action videos; torchvision reports
  63.2% top-1 accuracy on its 400 classes for these weights.
- **It is used as published.** Clip length, frame rate and the number of clips follow the settings the weights
  were evaluated with (16 frames at 15 fps, 5 clips per video), and the input size and normalisation are read
  from the weights.
- **It stays frozen.** Only a new linear head with 2,052 parameters is trained, on cached 512-dimensional clip
  features, which takes seconds. The 33.2 million backbone parameters are never updated.

The cost is compute. One 16-frame clip takes about 40.7 GFLOPs, against 0.057 GFLOPs for one frame through
MobileNetV3-Small, so with five clips per video the arithmetic is roughly 200 times that of the frame pipeline.
In measured time on this CPU the difference is smaller, about 2.0 s per video against 0.11 s for the backbone.

Fine-tuning the last block of the backbone was to be tried only if validation results left room for it. They
did not (section 7.1), so it was not run.

### 3.5 Training-only augmentation

Each training video is additionally cached in four augmented versions: a randomly positioned crop instead of
the centre crop, jittered frame or clip positions, brightness scaled by a factor between 0.8 and 1.2, and a
mirror image half of the time. Mirroring is used because all four exercises are left-right symmetric; it must
be switched off for classes defined by direction. Validation and test videos are never augmented. For the SVM
and the LSTM the augmented variant is a separate model; for the R3D-18 head, whether to use the augmented clips
is one of the choices made on validation.

### 3.6 Optical flow and learned temporal features

Optical flow estimates the displacement of image points between two frames. It is computed by a fixed algorithm
(Farneback for dense flow, Lucas-Kanade for sparse flow) with no training, and its output is a motion field that
can be inspected directly.

The LSTM works on CNN features, which describe what is visible in each frame. It can only detect motion
indirectly, as a change in appearance features between sampled frames (about seven frames apart on average
here), and it learns which changes matter from labelled data. R3D-18 sits in between: it learns motion-sensitive
filters from data and applies them to consecutive frames.

Flow and appearance features are complementary in principle, which is the idea behind two-stream action
recognition. Section 7.7 tests whether adding flow features helps here.

## 4. FOCV syllabus topics and where they are demonstrated

| Syllabus topic | Implementation | Role | App section | Code |
|---|---|---|---|---|
| Image transformations | Frame and clip sampling, resizing, cropping, normalisation | Classifier pipeline | Video Analysis | `video.py` |
| Image transformations | Random crops, brightness changes, mirroring | Training augmentation | none | `augment.py` |
| Image preprocessing | Gaussian and median filtering (with added salt-and-pepper noise to show the difference) | Demonstration | Image Processing | `image_processing.py` |
| Image preprocessing | Global, Otsu and adaptive thresholding; erosion, dilation, opening, closing, gradient | Demonstration | Image Processing | `image_processing.py` |
| Edge detection and features | Canny edge maps; edges as image structure and their relation to early CNN filters | Demonstration | Image Processing | `image_processing.py` |
| Object and shape analysis | Connected-component labelling with area, bounding box and centroid; contour tracing | Demonstration | Image Processing | `image_processing.py` |
| Motion analysis | Dense flow as a colour map and as motion vectors, sparse corner tracking, flow magnitude over time | Demonstration | Motion Analysis | `optical_flow.py` |
| Motion analysis | Flow histograms fused with CNN features | Separate experiment | Model Comparison | `motion_experiment.py` |
| CNN | MobileNetV3-Small as a spatial feature extractor; R3D-18 as a spatio-temporal one | Classifier pipeline | Activity Recognition | `features.py`, `r3d.py` |
| RNN | LSTM over successive frame features | Classifier pipeline | Activity Recognition | `lstm.py` |
| Classification and evaluation | All models compared on the same test videos | Classifier pipeline | Model Comparison, Evaluation | `baseline.py`, `evaluation.py` |

**Which methods support the classifiers.** Only sampling, resizing, cropping, normalisation and the networks
lie on the path from a video to a prediction, with augmentation added during training.

**Which are independent demonstrations.** Filtering, thresholding, morphology, Canny edges, contours and
connected components are shown on single frames to illustrate the course topics. They are deliberately not fed
to the classifiers: the networks were pretrained on unfiltered images, and a threshold mask of a natural video
frame isolates bright regions rather than people. Optical flow is a demonstration and a separate experiment; it
is not in the prediction pipeline because it did not improve validation results.

## 5. Dataset and split methodology

**Dataset.** UCF101 (Soomro, Zamir and Shah, 2012) contains 13,320 clips of 101 action classes collected from
YouTube. The project uses four classes: BodyWeightSquats, JumpingJack, PullUps and PushUps (437 clips, 320 x 240,
25 or 29.97 fps, 29 to 267 frames each, 107 on average). These body-weight exercises were chosen because they are
whole-body human activities filmed in similar indoor settings, which makes the scene less of a giveaway than it
would be for, say, swimming against archery.

The clips were downloaded per class from a third-party copy of UCF101 on Hugging Face (`sumith2425/UCF101`)
rather than from the 6.9 GB official archive. File sizes and SHA-256 checksums were verified against the host's
metadata. The copy was not compared with the official archive.

**Splitting by source video.** UCF101 clips named `v_<Class>_gXX_cYY` with the same group number were cut from
one source video. Splitting such clips between training and testing would let a model score well by recognising
the background or the person. The split therefore assigns whole groups:

| Split | Groups | Source videos | Clips | Use |
|---|---|---|---|---|
| Training | 8, 10, 11, 12, 13, 14, 16, 17, 18, 21, 22, 23, 24, 25 | 56 | 237 | fitting model weights |
| Validation | 9, 15, 19, 20 (drawn with seed 42) | 16 | 75 | choosing hyperparameters, early stopping |
| Test | 1 to 7 (official UCF101 split 1) | 28 | 125 | final evaluation |

| Class | Train | Validation | Test |
|---|---|---|---|
| BodyWeightSquats | 61 | 21 | 30 |
| JumpingJack | 64 | 22 | 37 |
| PullUps | 56 | 16 | 28 |
| PushUps | 56 | 16 | 30 |

**Audit.** `python -m visionact diagnose` found: all 437 files decode; no file name contradicts its class
folder; no two files are byte-identical; no near-duplicate videos are shared between splits (clips of one source
video have a median feature similarity of 0.974, the most similar pair from different source videos 0.753);
none of the 100 source videos appears in more than one split; training classes are balanced within a ratio of
1.14; refitting with shuffled training labels gives chance-level validation accuracy (24.5% and 25.2% for the
two feature sets, against 92.0% and 98.7% with the true labels). Frame rate coincides with class
(BodyWeightSquats and PushUps are 25 fps, the other two 29.97 fps); the models are not given the frame rate.

Features are computed per video, so frames of one video cannot be spread across splits. The code asserts on
every load that no source video appears in more than one split, and each model's `training.json` stores the
SHA-256 of the split it was trained on.

## 6. Experimental setup

**Hardware and software.** Intel Core i5-1235U laptop (10 cores), 7.7 GB RAM, no GPU, Windows 11, Python 3.14,
PyTorch 2.14.1 (CPU build), OpenCV 5.0, scikit-learn 1.9.1.

**Model selection.** All choices used the validation split only.

- *SVM:* the regularisation constant C was chosen from eight values between 0.0001 and 10 by validation
  macro-F1. C = 0.0001 was selected (0.0003 for the augmented variant).
- *LSTM:* twelve combinations of hidden size (64, 128, 256), dropout (0.3, 0.5) and learning rate (0.001,
  0.0003), each with early stopping on validation loss (patience 10 epochs). Selected: 128 hidden units,
  dropout 0.3, learning rate 0.001, weights of epoch 12 (for the augmented variant: learning rate 0.0003,
  epoch 7).
- *R3D-18 head:* eight combinations of augmented clips (yes, no), dropout (0, 0.5) and weight decay (0.0001,
  0.01), each with early stopping on validation loss (patience 20 epochs). Selected: augmented clips, no
  dropout, weight decay 0.0001, weights of epoch 67.
- *Default model for predictions:* the model with the best validation macro-F1, which is the R3D-18 head.

The test split was scored after these choices were fixed. No model was changed in response to a test result.

**Metrics.** Accuracy, per-class precision, recall and F1, macro F1, the confusion matrix, an exact McNemar test
on paired predictions, CPU inference time, training time and file sizes. Accuracy intervals are computed by
resampling source videos (a group bootstrap), because clips of one source video are not independent trials.

## 7. Results

### 7.1 Overall

125 test videos from 28 source videos. Chance level is 25%.

| Model | Test accuracy | 95% interval | Macro precision | Macro recall | Macro F1 | Validation accuracy |
|---|---|---|---|---|---|---|
| CNN + mean pooling + linear SVM | 77.6% (97/125) | 63.0% to 91.2% | 0.778 | 0.782 | 0.775 | 92.0% |
| CNN + LSTM | 74.4% (93/125) | 59.0% to 89.7% | 0.753 | 0.754 | 0.745 | 88.0% |
| CNN + SVM, augmented training | 75.2% (94/125) | 59.5% to 89.6% | 0.770 | 0.762 | 0.755 | 92.0% |
| CNN + LSTM, augmented training | 77.6% (97/125) | 62.2% to 92.7% | 0.788 | 0.789 | 0.777 | 86.7% |
| R3D-18 (frozen) + linear head | 100.0% (125/125) | 97.0% to 100% * | 1.000 | 1.000 | 1.000 | 98.7% |

\* With no errors, resampling cannot express uncertainty; this interval treats the videos as independent and is
too narrow. See section 9.

### 7.2 Per class

| Class (test videos) | SVM precision | SVM recall | LSTM precision | LSTM recall | R3D-18 precision | R3D-18 recall |
|---|---|---|---|---|---|---|
| BodyWeightSquats (30) | 0.767 | 0.767 | 0.733 | 0.733 | 1.000 | 1.000 |
| JumpingJack (37) | 0.862 | 0.676 | 0.846 | 0.595 | 1.000 | 1.000 |
| PullUps (28) | 0.688 | 0.786 | 0.622 | 0.821 | 1.000 | 1.000 |
| PushUps (30) | 0.794 | 0.900 | 0.812 | 0.867 | 1.000 | 1.000 |

Scores for the augmented variants are in `results/<model>/metrics.json`.

### 7.3 Confusion matrices

Rows are true classes, columns are predictions.

CNN + SVM:

| True \ Predicted | BodyWeightSquats | JumpingJack | PullUps | PushUps |
|---|---|---|---|---|
| BodyWeightSquats | 23 | 2 | 2 | 3 |
| JumpingJack | 4 | 25 | 8 | 0 |
| PullUps | 0 | 2 | 22 | 4 |
| PushUps | 3 | 0 | 0 | 27 |

CNN + LSTM:

| True \ Predicted | BodyWeightSquats | JumpingJack | PullUps | PushUps |
|---|---|---|---|---|
| BodyWeightSquats | 22 | 3 | 3 | 2 |
| JumpingJack | 4 | 22 | 11 | 0 |
| PullUps | 0 | 1 | 23 | 4 |
| PushUps | 4 | 0 | 0 | 26 |

R3D-18 + linear head: all 125 videos on the diagonal (30, 37, 28, 30).

The two frame-based models make the same kinds of error. The most frequent is JumpingJack predicted as PullUps;
in both activities the person is upright with raised arms. The figures are saved as
`results/<model>/confusion_matrix.png`.

### 7.4 Are the differences real?

Each model against the baseline on the same test videos (exact McNemar test):

| Model | Both correct | Only the baseline correct | Only this model correct | Both wrong | p |
|---|---|---|---|---|---|
| CNN + LSTM | 92 | 5 | 1 | 27 | 0.22 |
| CNN + SVM, augmented | 93 | 4 | 1 | 27 | 0.38 |
| CNN + LSTM, augmented | 93 | 4 | 4 | 24 | 1.00 |
| R3D-18 + linear head | 97 | 0 | 28 | 0 | below 0.001 |

The four MobileNet-based models cannot be told apart: they differ on at most eight videos and their intervals
overlap almost completely. R3D-18 is correct on every video the baseline gets right and on all 28 it gets wrong.

### 7.5 Do the LSTMs use temporal order?

Each trained LSTM was tested again on the same videos with the frame order destroyed.

| Model | Original order | Reversed | Shuffled |
|---|---|---|---|
| CNN + LSTM | 74.4% | 76.8% | 76.8% |
| CNN + LSTM, augmented training | 77.6% | 76.8% | 77.6% |

Accuracy does not fall when the order is destroyed. The LSTMs have not learned to rely on the order of the
frames; on this data they act as further appearance classifiers.

### 7.6 Why the frame-based models are limited

The diagnostics (which use training and validation data only) explain the 75% to 78% plateau.

- **Overfitting to the training source videos.** Every model, including the SVM, classifies 100% of the
  training clips correctly. The LSTM's training loss at its best epoch is 0.007 against a validation loss of
  0.459.
- **Errors arrive in whole source videos.** Of the 28 test source videos the baseline gets every clip right in
  19, every clip wrong in 4, and is mixed in 5.
- **Accuracy is limited by the number of source videos.** Accuracy on held-out source videos, averaged over
  random draws:

| Training source videos per class | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3 features | 42.8% | 58.4% | 65.5% | 71.3% | 71.5% | 71.9% | 74.3% | 77.7% |
| R3D-18 features | 71.5% | 87.6% | 96.0% | 97.3% | 98.1% | 98.1% | 98.5% | 98.5% |

- **Validation on 4 source videos per class is unreliable.** Over 200 random choices of the validation videos,
  accuracy with MobileNetV3 features averages 78.7% and ranges from 62.3% to 92.0% (5th to 95th percentile).
  The 92.0% obtained on the fixed validation split is at the top of that range, which is why it overstated the
  test result. With R3D-18 features the same experiment averages 98.9% (95.6% to 100%).

In short, ImageNet frame features need far more distinct scenes than 14 per class to separate the activity
from its setting, while features learned from action videos separate these classes with a handful of examples.

### 7.7 Optional experiment: optical-flow fusion

Dense Farneback flow was computed on 8 pairs of consecutive frames per video and summarised as histograms of
flow direction (72 numbers per video, about 41 ms per video). Linear SVMs were trained and selected on the same
splits, for each visual feature set with and without the flow descriptor.

| Features | Input size | Validation accuracy | Test accuracy |
|---|---|---|---|
| Optical-flow histograms only | 72 | 68.0% | 58.4% |
| MobileNetV3 mean pooling (the baseline) | 1024 | 92.0% | 77.6% |
| MobileNetV3 mean pooling + flow | 1096 | 92.0% | 79.2% |
| R3D-18 mean pooling | 512 | 98.7% | 100.0% |
| R3D-18 mean pooling + flow | 584 | 98.7% | 98.4% |

Motion alone reaches 58.4% against a chance level of 25%, so the flow descriptor carries information about the
activity. The rule for adopting it, fixed beforehand, was at least two more correct validation videos and a
higher validation macro-F1. Adding flow changed validation accuracy by zero videos for both feature sets, so
flow fusion is not part of the prediction pipeline. The test figures agree that nothing was lost: two videos
gained with MobileNetV3 features, two lost with R3D-18 features.

### 7.8 Cost on CPU

| Quantity | CNN + SVM | CNN + LSTM | R3D-18 + head |
|---|---|---|---|
| Backbone parameters (frozen) | 1.52 M | 1.52 M | 33.2 M |
| Trained parameters | 4 x 1024 weights | 591,364 | 2,052 |
| Feature extraction, whole dataset | 109 s | 109 s | 29 min (including augmented clips) |
| Training, selected model | 0.02 s | 6.7 s | 12.6 s |
| Training, all candidates | 0.8 s (8 fits) | 118 s (12 runs) | 56 s (8 runs) |
| Inference per video | 0.16 s | 0.14 s | 2.0 s |
| Size with backbone | 6.3 MB | 8.6 MB | 132.8 MB |

Inference times are medians over 20 test videos and are dominated by the backbone; the classifiers take 1 to
2 ms. The MobileNet figures differ only by measurement noise, since those models share everything but the
classifier. The R3D-18 extraction time was measured while other jobs were running. The augmented variants add
142 s of feature extraction; the augmented LSTM search took 399 s.

## 8. Discussion

**Does the LSTM improve on averaging?** No. On this dataset the LSTM is indistinguishable from the order-blind
baseline, with and without augmentation, and it does not use temporal order.

**What does improve recognition?** The representation. A frozen 3D CNN pretrained on action videos, with only
a linear layer trained on top, classified every test video correctly, and the hold-out analysis of the
training and validation data shows the same gap (78.7% against 98.9%) without reference to the test set.

**Why.** The four activities differ in body configuration and movement. ImageNet frame features capture the
room, the equipment and the person as much as the pose, so a classifier trained on 14 source videos per class
partly learns the scenes; this is visible in perfect training accuracy, errors that come in whole source
videos, and a learning curve still rising at 14 source videos. R3D-18 was trained to tell hundreds of actions
apart across thousands of scenes, including push-ups, pull-ups and squats, so its features already separate
these classes.

**What this does not show.** It does not show that temporal modelling is unnecessary in general. R3D-18 models
time inside its convolutions; the LSTM result says only that an LSTM over sparse ImageNet features adds
nothing here. Nor does a perfect score on 28 source videos mean the model is error-free.

## 9. Limitations

- **Small test set.** 125 clips from 28 source videos. For the frame-based models the 95% intervals are about
  30 points wide. For R3D-18, with all 28 source videos classified perfectly, the 95% lower bound on the share
  of source videos that would be classified perfectly is about 88%.
- **Pretraining overlap.** Kinetics-400 contains "push up", "pull ups" and "squat" classes (no jumping-jack
  class), so R3D-18 had been trained to recognise three of the four activities. Whether any Kinetics video
  shares a YouTube source with a UCF101 test clip was not checked.
- **Four classes, closed set.** Every classifier answers with one of its four classes, whatever the input, and
  none was tested on other activities.
- **Unreliable validation.** Four validation source videos per class make model selection noisy.
- **Repeated use of the test set.** Five models and five SVM variants were scored on it. No selection used test
  results, but small differences among the frame-based models should not be read as rankings.
- **Single seed, single split.** Variation across random seeds and across the three official UCF101 splits was
  not measured.
- **Dataset bias.** UCF101 clips of a class often share settings and camera angles. Accuracy on other footage
  (different cameras, viewpoints, lighting, people) is unknown.
- **Third-party data copy.** The clips came from a Hugging Face copy of UCF101, not the official archive.
- **Scores are not probabilities.** The softmax scores shown in the app are not calibrated.
- **CPU cost.** R3D-18 takes about 2 s per video on this hardware.
- **Not validated for real-world use.** The system is a teaching project and must not be relied on for safety,
  health monitoring, surveillance or similar purposes.

## 10. Possible future improvements

These are untested ideas, listed in rough order of expected value.

1. **A harder and larger test.** More classes, including ones absent from Kinetics-400 and ones that resemble
   each other, and all three official UCF101 splits with the mean reported.
2. **Repeat over seeds** and report mean and standard deviation.
3. **Classes that need temporal order,** such as sitting down against standing up, to give the LSTM a task
   where order matters.
4. **Group cross-validation for model selection** in place of one small validation split.
5. **Fine-tuning the last R3D-18 block,** once a harder task leaves room for improvement.
6. **Fewer clips or a lighter video model** to bring inference time down.
7. **Rejecting unknown inputs:** a confidence threshold or a background class so the app can answer "none of
   the known activities".

## 11. Reproducing the results

```
python -m visionact download
python -m visionact all
python -m visionact diagnose
python -m visionact motion-experiment
```

A second complete run on the development machine reproduced the per-video test predictions of all five models
exactly. Timings vary between runs. Outputs:

| File | Content |
|---|---|
| `artifacts/splits/manifest.csv`, `split_config.json` | the split and its configuration |
| `results/comparison.json`, `comparison.md` | sections 7.1 to 7.5 and 7.8 |
| `results/diagnostics.json` | the audit in section 5 and section 7.6 |
| `results/motion_experiment.json` | section 7.7 |
| `results/<model>/metrics.json` | accuracy, macro scores, per-class scores, confusion matrix |
| `results/<model>/predictions.csv` | true label, predicted label and scores for each test video |
| `results/<model>/training.json` | validation results for every candidate, selected settings, preprocessing, augmentation, split fingerprint, training history |
| `results/<model>/confusion_matrix.png`, `training_curves.png` | figures |
| `artifacts/models/` | trained checkpoints |

## References

- K. Soomro, A. R. Zamir and M. Shah. "UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild."
  CRCV-TR-12-01, 2012.
- A. Howard et al. "Searching for MobileNetV3." ICCV, 2019.
- S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory." Neural Computation, 1997.
- D. Tran et al. "A Closer Look at Spatiotemporal Convolutions for Action Recognition." CVPR, 2018.
- W. Kay et al. "The Kinetics Human Action Video Dataset." arXiv:1705.06950, 2017.
- G. Farnebäck. "Two-Frame Motion Estimation Based on Polynomial Expansion." SCIA, 2003.
- B. D. Lucas and T. Kanade. "An Iterative Image Registration Technique with an Application to Stereo Vision."
  IJCAI, 1981.
- J. Canny. "A Computational Approach to Edge Detection." IEEE TPAMI, 1986.
- K. Simonyan and A. Zisserman. "Two-Stream Convolutional Networks for Action Recognition in Videos." NeurIPS, 2014.
