#!/bin/bash

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/d4rl/mlp_ebm_langevin_best2.gin \
  --task=LANG_ROBOT \
  --tag=ibc_dfo \
  --add_time=True \
  --skip_eval=True \
  --saving_folder=awo_testin_bigbs_lr3 \
   --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single/tw_data*.tfrecord'" \
  --gin_bindings="train_eval.batch_size=512" \
  --gin_bindings="MLPEBM.width=512" \
  --gin_bindings="MLPEBM.depth=8" \
  --gin_bindings="train_eval.learning_rate=5e-4" \
  $@
  #--gin_bindings="train_eval.dataset_path='ibc/data/UR5_single_smollest/tw_data*.tfrecord'" \
  # --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single_smol/tw_data*.tfrecord'" \
  # --gin_bindings="train_eval.dataset_path='ibc/data/UR5_single/tw_data*.tfrecord'" \
  #--gin_bindings="train_eval.dataset_path='ibc/data/UR5/tw_data*.tfrecord'" \
  # --video
