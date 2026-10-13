# VisionAct improvement report

What limited the original accuracy, what was changed, and what each change measurably did. Every figure comes
from the files in `results/`, produced with seed 42 by `python -m visionact all`, `diagnose` and
`motion-experiment`.

## 1. Summary

- The original models reached 77.6% (CNN + SVM) and 74.4% (CNN + LSTM) test accuracy. The data and the split
  were sound; the cause was features that did not carry over to new source videos.
- **Largest measured improvement:** replacing ImageNet frame features with a frozen R3D-18 pretrained on
  Kinetics-400 and training a new linear head. Test accuracy rose from 77.6% to 100.0% (125 of 125 videos), and
  accuracy over repeated hold-outs of the training and validation data rose from 78.7% to 98.9%.
- Training-time augmentation, the LSTM and optical-flow fusion produced no reliable gain.
- The 100% figure needs care: it rests on 28 source videos and on a backbone that had already been trained to
  recognise three of the four activities.

## 2. Diagnosis of the original accuracy

`python -m visionact diagnose` re-scores the saved models and audits the data. Its experiments use training and
validation videos only.

### What was ruled out

| Possible cause | Check | Result |
|---|---|---|
| Unreadable or mislabelled files | Decode every file; compare file name with class folder | 437 of 437 decode, 0 mismatches |
| Leakage between splits | Each source video (class and group) must be in one split | 100 source videos, 0 in more than one split |
| Duplicate videos | SHA-256 of every file | 0 identical files |
| Near-duplicate videos across splits | Similarity of pooled CNN features | 0 suspicious pairs: clips of one source video have median similarity 0.974, the most similar pair from different source videos 0.753 |
| Class imbalance | Training videos per class | 56 to 64, ratio 1.14 |
| Inconsistent preprocessing | Re-extract features with the inference code and compare with the cache | identical |
| Results not from the saved models | Re-score the checkpoints | stored test predictions reproduced exactly |
| Accuracy not coming from the labels | Refit with shuffled training labels, 100 times | 24.5% on average against 92.0% with true labels (chance 25%) |

One oddity was found and is harmless here: frame rate coincides with class (BodyWeightSquats and PushUps are
25 fps, JumpingJack and PullUps 29.97 fps). The models never see the frame rate.

### What was found

1. **Overfitting to the training source videos.** Both models classified every training clip correctly (100%)
   but only 77.6% and 74.4% of test clips. The LSTM's training loss was 0.007 at its best epoch against a
   validation loss of 0.459.
2. **Too few independent training examples for these features.** Training has 237 clips but only 14 source
   videos per class, and clips of one source video are near-copies. Accuracy on held-out source videos was still
   climbing when the data ran out: 71.9% with 10 training source videos per class, 77.7% with 14.
3. **Errors come in whole source videos.** Of the 28 test source videos, the baseline got every clip right in 19,
   every clip wrong in 4, and was mixed in 5; 79% of its errors sit in source videos it mostly failed. The test
   set therefore behaves like 28 samples, not 125. Resampling source videos gives a 95% interval of 63.0% to
   91.2% for the baseline, against the 69.5% to 84.0% reported originally.
4. **The validation score was not reliable.** With 4 validation source videos per class, accuracy ranged from
   62.3% to 92.0% (5th to 95th percentile) over 200 random choices of validation videos, with a mean of 78.7%.
   The original 92.0% was a lucky draw; the mean matches the test result. The apparent drop from validation to
   test was therefore mostly sampling noise, and hyperparameter selection on this split is noisy.

**Conclusion.** Neither the classifier (SVM or LSTM) nor the amount of tuning was the limit. The limit was the
representation: ImageNet features of single frames describe scenes and people, and 14 scenes per class are too
few to learn what the activity itself looks like.

## 3. What was changed

| Change | Purpose |
|---|---|
| R3D-18 pretrained on Kinetics-400, frozen, with a new linear head (`r3d.py`) | Features learned from action videos that describe motion as well as appearance |
| Clip sampling as the weights expect: 16 frames at about 15 fps, 5 clips per video, logits averaged | Correct use of the pretrained model |
| Training-only augmentation (`augment.py`): random crop position, temporal jitter, brightness 0.8 to 1.2, mirroring | More varied training data; cached as 4 extra copies per training video |
| Augmented variants of the SVM and the LSTM, saved as separate models | Measure augmentation without altering the original baselines |
| Evaluation of every model through one registry, on the same test videos | Fair comparison |
| Intervals that resample source videos; per-source-video error breakdown | Honest uncertainty |
| Experiment record in every `training.json` (split fingerprint, preprocessing, augmentation, hyperparameters) | Reproducibility |
| `diagnose` command | Repeatable data audit and leakage checks |
| Optical-flow fusion experiment with a validation-based decision rule | Test whether motion features add anything |

The two original models were retrained unchanged and reproduce their earlier test predictions exactly.

**Not done, and why.** Fine-tuning the last R3D-18 block was planned only if validation left room for it. The
frozen model already classifies 74 of 75 validation videos correctly, so no improvement could have been
measured, and the experiment was not run.

## 4. Measured results

125 test videos from 28 source videos. Each model was selected on validation data only.

| Model | Test accuracy | 95% interval | Macro F1 | Validation accuracy | Training | Inference per video | Size with backbone |
|---|---|---|---|---|---|---|---|
| CNN + mean pooling + linear SVM (baseline) | 77.6% (97/125) | 63.0% to 91.2% | 0.775 | 92.0% | 0.02 s | 0.16 s | 6.3 MB |
| CNN + LSTM | 74.4% (93/125) | 59.0% to 89.7% | 0.745 | 88.0% | 6.7 s | 0.14 s | 8.6 MB |
| CNN + SVM, augmented training | 75.2% (94/125) | 59.5% to 89.6% | 0.755 | 92.0% | 0.4 s | 0.13 s | 6.3 MB |
| CNN + LSTM, augmented training | 77.6% (97/125) | 62.2% to 92.7% | 0.777 | 86.7% | 24.3 s | 0.13 s | 8.6 MB |
| R3D-18 (frozen) + linear head | 100.0% (125/125) | 97.0% to 100% * | 1.000 | 98.7% | 12.6 s | 2.0 s | 132.8 MB |

Intervals resample whole source videos. \* With no errors that method cannot show uncertainty, so this interval
treats the 125 videos as independent and is too narrow; see section 7.

Training times are for the selected configuration on cached features. Extracting the features is the larger
cost: about 2 minutes for MobileNetV3-Small on all videos, and 29 minutes for R3D-18 (3,133 clips, measured
while other jobs were running). Per-class scores and confusion matrices are in `results/<model>/`.

Comparison with the baseline on the same test videos:

| Model | Only the baseline correct | Only this model correct | McNemar p |
|---|---|---|---|
| CNN + LSTM | 5 | 1 | 0.22 |
| CNN + SVM, augmented | 4 | 1 | 0.38 |
| CNN + LSTM, augmented | 4 | 4 | 1.00 |
| R3D-18 + linear head | 0 | 28 | below 0.001 |

## 5. Largest measured improvement

**The pretrained video backbone.** Nothing else came close.

| Evidence | MobileNetV3 features | R3D-18 features |
|---|---|---|
| Test accuracy | 77.6% | 100.0% |
| Test videos gained or lost against the baseline | reference | 28 gained, 0 lost |
| Validation accuracy (fixed split) | 92.0% | 98.7% |
| Mean accuracy over 200 random hold-outs of training and validation data | 78.7% | 98.9% |
| Range of those hold-outs (5th to 95th percentile) | 62.3% to 92.0% | 95.6% to 100% |
| Accuracy with only 2 training source videos per class | 58.4% | 87.6% |
| Accuracy with 4 training source videos per class | 65.5% | 96.0% |

The hold-out rows do not involve the test set, so the improvement is not an accident of which videos happened to
be in it. R3D-18 features need far fewer training examples, which is the direct remedy for the problem found in
section 2. The model is also stable: its validation loss is 0.034 against 0.459 for the LSTM.

The price is compute: about 2.0 s per video instead of 0.15 s, a 133 MB backbone instead of 6 MB, and roughly
half an hour of one-time feature extraction on this CPU.

## 6. What did not help

- **Augmentation.** The SVM lost 3 test videos with it and the LSTM gained 4; neither change is distinguishable
  from noise, and validation accuracy did not improve for either. For the R3D-18 head, the selected candidate
  used the augmented clips (74 of 75 validation videos against 73 of 75 without), a difference of one video.
  Augmented copies of 14 source videos per class are still the same 14 scenes.
- **The LSTM.** With or without augmentation it is statistically indistinguishable from mean pooling, and
  shuffling the frames at test time does not lower its accuracy (74.4% in order, 76.8% shuffled; 77.6% and
  77.6% for the augmented variant). It does not use temporal order on this data.
- **Optical-flow fusion.** Flow histograms alone reach 58.4% on the test set (68.0% on validation), so they
  carry real information. Added to the visual features they changed validation accuracy by 0 videos with both
  MobileNetV3 and R3D-18 features. The rule fixed in advance required at least 2 more correct validation videos
  and a higher macro-F1, so flow fusion is not part of the prediction pipeline. On the test set it moved the
  MobileNetV3 result from 77.6% to 79.2% and the R3D-18 result from 100% to 98.4%. The flow visualisations
  remain in the app.

## 7. Remaining limitations

- **The 100% is a small-sample result.** The 125 test clips come from 28 source videos. With all 28 classified
  perfectly, a 95% lower bound on the share of source videos that would be classified perfectly is about 88%.
  Expect errors on more data.
- **The backbone already knew most of these activities.** Kinetics-400 has "push up", "pull ups" and "squat"
  classes (no jumping-jack class). The result shows that knowledge transferring to UCF101 clips; it does not
  show the model learning an activity from 14 examples. Whether any Kinetics training video shares a YouTube
  source with a UCF101 test clip was not checked.
- **Four visually distinct classes.** Nothing here says how the models would do on many classes or on similar
  ones.
- **Closed set.** Every input is assigned one of the four classes, including videos of other activities.
- **Noisy selection.** The validation set is too small to choose reliably between models that are close.
  This affected the MobileNet-based models; the R3D-18 result is far outside the noise.
- **Repeated use of the test set.** Five models, and five SVM variants in the flow experiment, were scored on
  the same test videos. No selection used test results, but the small differences among the MobileNet-based
  models should not be read as rankings.
- **Single seed, single split.** Variation across seeds and across the three official UCF101 splits was not
  measured.
- **Third-party data copy.** The clips came from a Hugging Face copy of UCF101, not the official archive.
- **CPU cost.** R3D-18 needs about 2 s per video here, which rules out real-time use on this hardware.
- **Not validated for real-world use.** Performance on other cameras, viewpoints, lighting and people is unknown.

## 8. Reproducing

```
python -m visionact download
python -m visionact all
python -m visionact diagnose
python -m visionact motion-experiment
```

A second complete run on the development machine reproduced the test predictions of all five models exactly.
Timings vary between runs.

| File | Content |
|---|---|
| `results/comparison.json`, `comparison.md` | sections 4 and 5 |
| `results/diagnostics.json` | section 2 and the hold-out rows of section 5 |
| `results/motion_experiment.json` | optical-flow fusion |
| `results/<model>/training.json` | candidates, selected hyperparameters, preprocessing, augmentation, split fingerprint |
| `results/<model>/predictions.csv`, `metrics.json`, `confusion_matrix.png` | per-video predictions and metrics |
| `artifacts/models/` | trained checkpoints |
