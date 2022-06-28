#!/bin/bash

# Use name of d4rl env as first arg

CMD='python3 ibc/ibc/train_eval.py '
GIN='ibc/ibc/configs/tw/mlp_ebm_langevin_best.gin'
#DATA="train_eval.dataset_path='ibc/data/d4rl_trajectories/$1/*.tfrecord'"

$CMD -- \
  --alsologtostderr \
  --gin_file=$GIN \
  --task=LANG_ROBOT \
  --tag=ibc_langevin \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single/tw_data*.tfrecord'" \
  # not currently calling --video because rendering is broken in the docker?
