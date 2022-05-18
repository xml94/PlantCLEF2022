#!/bin/bash
####################
# train from a given epoch
####################
export IMAGENET_DIR='/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022/'
export name="IN1k_Clef2022"
export all_epoch=100
export resume_epoch=49
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_large_patch16  \
    --epochs ${all_epoch} \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --log_dir checkpoint/${name}/"log" \
    --nb_classes 80000 \
    --resume "./checkpoint/${name}/checkpoint-${resume_epoch}.pth" --start_epoch ${resume_epoch} \
    --eval_epoch 1 \
    --save_model_epoch 5


####################
# train from the first epoch
####################
#export IMAGENET_DIR='/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022/'
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_large.pth'
#export name="IN1k_Clef2022"
#export all_epoch=100
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
#    --accum_iter 4 \
#    --batch_size 32 \
#    --model vit_large_patch16  \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${all_epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --log_dir checkpoint/${name}/"log" \
#    --nb_classes 80000 \
#    --eval_epoch 1 \
#    --save_model_epoch 5