#!/bin/bash

lr=2e-5   # 2e-5
wd=0.05   # 0.1
dropout=0.05  # 0.05
z_loss_weight=1e-5

data_config="./pre_tokenization/mimic_impression_base/meta_mimic_impression_base.json"

exp_name=baseline
mkdir -p output/"$exp_name"
#export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=25640 finetune_solver_align.py \
--model_size 7B \
--batch_size 8 \
--accum_iter 1 \
--epochs 5 \
--warmup_epochs 0.01 \
--lr ${lr} \
--min_lr ${lr} \
--wd ${wd} \
--clip_grad 4 \
--data_config $data_config \
--cache_ann_on_disk \
--num_workers 8 \
--output_dir output/"$exp_name" \
--save_iteration_interval 1000 \
--ckpt_max_keep 2 \
--checkpointing \
--max_seq_len 4096 \
--unmask_image_logits \
--dropout ${dropout} \
--z_loss_weight ${z_loss_weight} \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"
