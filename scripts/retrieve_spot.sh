#!/bin/bash

DATA_PATH=your_data/spot-the-diff/download
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6666 main_task_retrieval.py \
--do_train \
--num_thread_reader=4 \
--epochs=20 \
--batch_size=128 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/images \
--output_dir ckpts/ckpt_spot_retrieval \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 128 \
--datatype spot \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32
