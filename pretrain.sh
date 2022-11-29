#export IMAGENET_DIR='/data/Mingle/datasets/PlantCLEF2022/'
#export PRETRAIN_CHKPT='./checkpoint/mae_pretrain_vit_large.pth'
#export name="IN1k_Clef2022"
#export all_epoch=100
#python3 main_finetune.py \
#    --accum_iter 1 \
#    --batch_size 1024 \
#    --model vit_large_patch16  \
#    --finetune ${PRETRAIN_CHKPT} \
#    --epochs ${all_epoch} \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#    --dist_eval --data_path ${IMAGENET_DIR} \
#    --log_dir "checkpoint/${name}/log" \
#    --nb_classes 80000 \
#    --eval_epoch 1 \
#    --save_model_epoch 10 \
#    --output_dir checkpoint/${name} \
#    --num_workers 16


# vit-large
export IMAGENET_DIR="/home/oem/Mingle/datasets/PlantCLEF2022/PlantCLEF2022_web"
export name="MAE_PlantCLEF2022_web_train_pretrain_vit_large"
#export name="MAE_PlantCLEF2022_trust_train_pretrain_vit_large"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
  --accum_iter 8 \
  --batch_size 128 \
  --model mae_vit_large_patch16 \
  --norm_pix_loss \
  --mask_ratio 0.75 \
  --epochs 800 \
  --warmup_epochs 40 \
  --blr 1.5e-4 --weight_decay 0.05 \
  --data_path ${IMAGENET_DIR} \
  --log_dir "checkpoint/${name}/log" \
  --output_dir checkpoint/${name}


# vit-base
export IMAGENET_DIR="/home/oem/Mingle/datasets/PlantCLEF2022/PlantCLEF2022_web"
export name="MAE_PlantCLEF2022_web_train_pretrain_vit_base"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
  --accum_iter 8 \
  --batch_size 128 \
  --model mae_vit_base_patch16 \
  --norm_pix_loss \
  --mask_ratio 0.75 \
  --epochs 800 \
  --warmup_epochs 40 \
  --blr 1.5e-4 --weight_decay 0.05 \
  --data_path ${IMAGENET_DIR} \
  --log_dir "checkpoint/${name}/log" \
  --output_dir checkpoint/${name}