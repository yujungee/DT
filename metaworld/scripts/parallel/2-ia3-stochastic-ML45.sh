#!/bin/bash

model_path='/home/chois/decision-transformer/metaworld/saved_model/pretrained/2023-07-26_16:01:52_ML45_/model_ML45_TRAIN_expert_STOCHASTIC_POLICY_iter_10'
task_list="
assembly-v2   basketball-v2   button-press-topdown-wall-v2   button-press-v2   button-press-wall-v2   coffee-button-v2   coffee-pull-v2   coffee-push-v2   dial-turn-v2   disassemble-v2
"

for var in $task_list
do
    python experiment-offline.py  --env $var --tuning_type ia3 --batch_size 64 --model_path $model_path --stochastic_policy --num_steps_per_iter 5000 --save_path "saved_model/ML45_ia3_tuning/" --eval_interval 10
done

# drawer-close-v2   reach-v2   window-close-v2   window-open-v2   button-press-topdown-v2   door-open-v2   drawer-open-v2   pick-place-v2   peg-insert-side-v2   push-v2
# door-close-v2   faucet-open-v2   faucet-close-v2   hammer-v2   handle-press-side-v2   handle-press-v2   handle-pull-side-v2   handle-pull-v2   lever-pull-v2   pick-place-wall-v2
# push-back-v2   plate-slide-v2   plate-slide-side-v2   plate-slide-back-v2   plate-slide-back-side-v2   peg-unplug-side-v2   soccer-v2   stick-push-v2   stick-pull-v2   push-wall-v2
# reach-wall-v2   shelf-place-v2   sweep-into-v2   sweep-v2
# bin-picking-v2 box-close-v2 door-lock-v2 door-unlock-v2 hand-insert-v2