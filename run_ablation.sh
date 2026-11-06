#!/bin/sh
# 2 training sets x 2 augmentation settings x 2 seeds, one job at a time.
export OMP_NUM_THREADS=4
for seed in 0 1; do
  for aug in default off; do
    for tr in rand fixed; do
      nice python train_eval.py ${tr}_${aug}_s${seed} data/${tr}_train --aug $aug --seed $seed > logs_train_${tr}_${aug}_s${seed}.log 2>&1
    done
  done
done
echo done > ablation.done
