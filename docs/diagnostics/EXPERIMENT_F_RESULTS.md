# Experiment F: longer training and higher input resolution

Both runs fine-tune Experiment E (`yolo26s_full_best.pt`) for 10 epochs on extended v3 with MuSGD, lr0 0.001,
lrf 0.1, warmup 0.5 epochs (`scripts/run_experiment_f.sh`). F1 trains and tests at 640 px (control: more epochs only);
F2 at 800 px. An earlier attempt with `optimizer=auto` was aborted (see `experiments/yolo26s_ft640_aborted_autoopt`).

## Ultralytics evaluation (mAP@50)

| Test split | E | F1 640 px | F2 800 px |
|---|---|---|---|
| RDD2022 test (2,023) | 62.4 % | 63.9 % | **64.2 %** |
| Clean test subset (1,804) | 61.8 % | 63.3 % | **63.6 %** |
| Norway + China_Drone (532) | 40.9 % | 41.5 % | **43.7 %** |
| Egypt + Roboflow potholes (221) | 58.8 % | 58.9 % | **59.8 %** |
| Close-up potholes (235) | **67.2 %** | 67.0 % | 66.3 % |
| Pavement-distress (1,803) | 59.0 % | **59.3 %** | **59.3 %** |
| Rome, new country (634, potholes) | 25.7 % | 24.5 % | **28.3 %** |

F2 on the RDD2022 test: precision 65.8 %, recall 58.8 %, F1 62.1 %, mAP@50-95 33.4 %.

## Paired bootstrap (1,000 image resamples, `scripts/bootstrap_ci.py`)

Values from our own matcher (101-point AP, IoU 0.5); absolute numbers are about 1 pp below Ultralytics', the
differences are what matter. Raw output: `docs/diagnostics/bootstrap/*.json`.

| Split | F1 − E (95 % CI) | F2 − E (95 % CI) | F2 − F1 (95 % CI) |
|---|---|---|---|
| RDD2022 test | +1.6 pp [+1.0, +2.2] | +2.4 pp [+1.5, +3.4] | +0.8 pp [+0.0, +1.8] |
| Clean subset | +1.5 pp [+0.9, +2.1] | +2.4 pp [+1.4, +3.3] | +0.9 pp [+0.0, +1.8] |
| Rome | −0.2 pp [−2.5, +2.2] | +3.8 pp [+0.9, +7.0] | +4.0 pp [+0.7, +7.6] |

## Conclusions

* Under-training was real: 10 more epochs gave a significant +1.5 pp in domain (F1), but nothing on Rome.
* Higher resolution helps small damage: F2 adds a further, borderline-significant +0.8 pp in domain, +2.2 pp on
  Norway/China_Drone and a significant +4.0 pp on Rome, where recall rose from 32.8 % to 43.9 %.
* Cost: 35.8 vs 22.8 GFLOPs; end-to-end latency on the RTX 3050 19.8 ms vs 14.1 ms per image (batch 1); same
  9.95 M parameters and 19.4 MB file.
