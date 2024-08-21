lambda_composition=1.0
lambda_oneshot=1.0
lambda_slot_diffusion=1.0
lambda_mask_reg=0.5
cnn_enc_type="unet"
steps=210000
port=$1
gpus=$2
latent_size=192
num_slots=5
diff_dim=192

LATENT_SIZE=${latent_size}
NUM_DEC_BLOCKS=8
MLP_SIZE=${latent_size}
D_TF=384
NUM_HEADS=8
subset_portion=1.0
ddim_steps=1
max_steps=500
lr_lsd=1e-4
lr_sa=1e-4
cnn_downsample=2
attn_iter=7

exp_name=msn/test_release
echo ${exp_name}
echo ${gpus}

accelerate launch --use_deepspeed --gradient_clipping 5.0 --num_processes 4 --main_process_port=${port} --gpu_ids=${gpus} train_object_discovery.py \
    --exp_name=${exp_name} \
    --num_slots ${num_slots} \
    --lr_sa ${lr_sa} \
    --lr_lsd ${lr_lsd} \
    --seed 1234 \
    --log_freq 5000 \
    --save_freq 50000 \
    --cnn_downsample ${cnn_downsample} \
    --dataset_name 'msn-easy' \
    --image_size 128 \
    --batch_size 16 \
    --lambda_composition ${lambda_composition} \
    --lambda_oneshot ${lambda_oneshot} \
    --lambda_slot_diffusion ${lambda_slot_diffusion} \
    --lambda_mask_reg ${lambda_mask_reg} \
    --latent_size ${LATENT_SIZE} \
    --input_channels 3 \
    --attention_iters ${attn_iter} \
    --mlp_size ${MLP_SIZE} \
    --num_dec_blocks ${NUM_DEC_BLOCKS} \
    --d_tf ${D_TF} \
    --num_heads ${NUM_HEADS} \
    --steps ${steps} \
    --cnn_enc_type ${cnn_enc_type} \
    --use_accel \
    --subset_portion ${subset_portion} \
    --log_n_imgs 4 \
    --slot_encode_RGB \
    --ddim_steps ${ddim_steps} \
    --max_steps ${max_steps} \
    --scale_latent 1.0 \
    --diff_dim ${diff_dim} \
    --share_slot_init 

