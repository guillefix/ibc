#!/bin/bash

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/d4rl/mlp_ebm_langevin_best2.gin \
  --task=LANG_ROBOT_LANG \
  --tag=ibc_dfo \
  --add_time=True \
  --skip_eval=True \
  --saving_folder=awo_testin_lang1f \
  --decay_steps=300 \
  --decay_rate=0.99 \
  --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single_lang/tw_data*.tfrecord'" \
  --gin_bindings="train_eval.batch_size=1024" \
  --gin_bindings="train_eval.learning_rate=1e-4" \
  --gin_bindings="train_eval.network='MLPEBMLang'" \
  $@
  #--gin_bindings="train_eval.dataset_path='ibc/data/UR5_single_smollest/tw_data*.tfrecord'" \
  # --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single_smol/tw_data*.tfrecord'" \
  # --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single/tw_data*.tfrecord'" \
  #--gin_bindings="train_eval.dataset_path='ibc/data/UR5/tw_data*.tfrecord'" \
  # --video
