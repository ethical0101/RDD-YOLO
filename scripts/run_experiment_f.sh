#!/usr/bin/env bash
# Experiment F: does input resolution help small damage? Controlled pair, both fine-tuned from Experiment E
# (yolo26s_full_best.pt) for the same epochs on the same data (extended v3) with the same settings:
#   F1 yolo26s_ft640 - 640 px, batch 8 (control: isolates the effect of "more epochs")
#   F2 yolo26s_ft800 - 800 px, batch 6 (Ultralytics accumulates gradients to nbs=64 in both)
# Optimiser: MuSGD as in Experiment E (optimizer=auto would switch a 10-epoch run to a fresh AdamW, which knocked an
# earlier attempt off E's optimum: val mAP@50 61.4 -> 49.8 %; kept as experiments/yolo26s_ft640_aborted_autoopt),
# lr0 0.001 (a tenth of E's), lrf 0.1, warmup 0.5 epochs.
# Never runs two GPU jobs at once. The served model is not changed.
set -e
cd "$(dirname "$0")/.."
P=.venv/Scripts/python; L=experiments/logs; E=dataset/processed/rdd2022_extended_v3; INIT=models/weights/yolo26s_full_best.pt
EPOCHS=${EPOCHS:-10}
run() { # name imgsz batch
  local n=$1 s=$2 b=$3
  if [ ! -f experiments/$n/run_info.json ]; then
    if [ -f experiments/$n/weights/last.pt ]; then
      $P training/train.py --resume experiments/$n/weights/last.pt >> $L/${n}_train.log 2>&1
    else
      $P training/train.py --experiment baseline --scale s --name $n --data $E/data.yaml --init-weights $INIT \
        --title "Experiment F - YOLO26s fine-tune at ${s}px" --imgsz $s --epochs $EPOCHS --batch $b --workers 6 \
        --patience 8 --close-mosaic 3 --non-deterministic --optimizer MuSGD --lr0 0.001 --lrf 0.1 --warmup-epochs 0.5 > $L/${n}_train.log 2>&1
    fi
  fi
  echo "$(date +%H:%M) $n trained"
  $P training/evaluate.py --run $n --data $E/data.yaml --split test --tag test --imgsz $s > $L/${n}_eval_test.log 2>&1
  for t in test_clean test_rdd_new test_external test_closeup test_pavement test_rome; do
    $P training/evaluate.py --run $n --data $E/data_$t.yaml --split test --tag $t --imgsz $s --speed-images 50 > $L/${n}_eval_$t.log 2>&1
  done
  echo "$(date +%H:%M) $n evaluated"
}
run yolo26s_ft640 640 8
run yolo26s_ft800 800 6
echo "$(date +%H:%M) ALL DONE"
