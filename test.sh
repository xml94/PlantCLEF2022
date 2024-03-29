##########################
#  ClefPlant2022
##########################
export IMAGENET_DIR="/home/oem/Mingle/datasets/ClefPlant2022_test/"
export name="IN1k_ClefPlant2022"
export epoch=100
CUDA_VISIBLE_DEVICES=0 python3 main_finetune.py \
--eval \
--resume "checkpoint/${name}/checkpoint-${epoch}.pth" \
--model vit_large_patch16 \
--batch_size 1 \
--data_path ${IMAGENET_DIR} \
--nb_classes 80000 \
--visualize_epoch "${epoch}" \
--max_num 30