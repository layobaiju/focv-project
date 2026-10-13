Test set: 125 videos, classes: BodyWeightSquats, JumpingJack, PullUps, PushUps.

| Model | Test accuracy | 95% interval | Macro F1 | Validation macro F1 | Training (s) | Inference (ms/video) | Size with backbone (MB) |
|---|---|---|---|---|---|---|---|
| CNN + mean pooling + linear SVM | 77.6% | 63.0% to 91.2% | 0.775 | 0.921 | 0.02 | 156 | 6.3 |
| CNN + LSTM | 74.4% | 59.0% to 89.7% | 0.745 | 0.881 | 6.72 | 135 | 8.6 |
| CNN + SVM, augmented training | 75.2% | 59.5% to 89.6% | 0.755 | 0.923 | 0.38 | 131 | 6.3 |
| CNN + LSTM, augmented training | 77.6% | 62.2% to 92.7% | 0.777 | 0.867 | 24.33 | 131 | 8.6 |
| R3D-18 (Kinetics-400, frozen) + linear head (best on validation) | 100.0% | 97.0% to 100.0% * | 1.000 | 0.986 | 12.59 | 1996 | 132.8 |

Intervals resample whole source videos, since clips of one source video succeed or fail together.
\* No errors, so resampling cannot show uncertainty; this interval treats the videos as independent and is therefore too narrow.
