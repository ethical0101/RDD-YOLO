#!/usr/bin/env bash
# Experiment E: YOLO26s (unmodified latest Ultralytics architecture, small scale), full COCO init,
# trained on the extended v3 dataset. Does NOT change the model served by the app.
set -e
cd "$(dirname "$0")/.."
P=.venv/Scripts/python; L=experiments/logs; E=dataset/processed/rdd2022_extended_v3
EPOCHS=${EPOCHS:-20}
until [ -f dataset/raw/external/deeksha/.complete ]; do sleep 30; done
[ -f $E/data.yaml ] || $P -u dataset/build_extended_v3.py > $L/build_extended_v3.log 2>&1
echo "$(date +%H:%M) dataset v3 built"
[ -f experiments/rdd_yolo26n_ext2/eval_test_pavement/metrics.json ] || \
  $P training/evaluate.py --run rdd_yolo26n_ext2 --data $E/data_test_pavement.yaml --split test --tag test_pavement > $L/rdd_yolo26n_ext2_eval_test_pavement.log 2>&1
echo "$(date +%H:%M) before-eval done"
if [ ! -f experiments/yolo26s_full/run_info.json ]; then
  if [ -f experiments/yolo26s_full/weights/last.pt ]; then
    $P training/train.py --resume experiments/yolo26s_full/weights/last.pt >> $L/yolo26s_full_train.log 2>&1
  else
    $P training/train.py --experiment baseline --scale s --name yolo26s_full --data $E/data.yaml \
      --title "Experiment E - YOLO26s, full COCO init, extended data v3" --init-all-layers \
      --epochs $EPOCHS --batch 8 --workers 6 --patience 8 --close-mosaic 5 --non-deterministic > $L/yolo26s_full_train.log 2>&1
  fi
fi
echo "$(date +%H:%M) training done"
$P training/evaluate.py --run yolo26s_full --data $E/data.yaml --split test --tag test > $L/yolo26s_full_eval_test.log 2>&1
for t in test_rdd_new test_external test_closeup test_pavement; do
  $P training/evaluate.py --run yolo26s_full --data $E/data_$t.yaml --split test --tag $t > $L/yolo26s_full_eval_$t.log 2>&1
done
echo "$(date +%H:%M) ALL DONE"
