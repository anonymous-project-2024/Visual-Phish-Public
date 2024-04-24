MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
SIAMESE_FLAGS="--siamese_path ./OCR_siamese_utils/output/targetlist_lr0.01/bit.pth.tar"
OCR_FLAGS="--ocr_path ./OCR_siamese_utils/demo_downgrade.pth.tar"
CHECKPOINT_PATH="save/logo_BOA/ema_0.9999_410000.pt"
OPENAI_LOGDIR=save/logo_BOA/samples/  python scripts/image_sample.py --num_samples 100 --model_path $CHECKPOINT_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $SIAMESE_FLAGS $OCR_FLAGS --timestep_respacing ddim100
