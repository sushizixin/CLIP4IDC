#!/bin/bash

DATA_PATH=your_data/clevr_change/data
python -m torch.distributed.launch --nproc_per_node=1 --master_port=5557 main_task_caption.py \
--do_eval \
--num_thread_reader=4 \
--epochs=50 \
--batch_size=16 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH} \
--output_dir ckpts/ckpt_clevr_caption_eval \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 32 \
--datatype clevr \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ckpts/trained/pytorch_model.bin.clevr
