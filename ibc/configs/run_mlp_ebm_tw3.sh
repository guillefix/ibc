#!/bin/bash

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/d4rl/mlp_ebm_langevin_best2.gin \
  --task=LANG_ROBOT \
  --tag=ibc_dfo \
  --add_time=True \
  --skip_eval=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single_smollest/tw_data*.tfrecord'" \
  --gin_bindings="train_eval.batch_size=64" \
  # --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single_smol/tw_data*.tfrecord'" \
  # --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single/tw_data*.tfrecord'" \
  #--gin_bindings="train_eval.dataset_path='ibc/data/UR5/tw_data*.tfrecord'" \
  # --video
