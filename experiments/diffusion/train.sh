MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 2 --save_interval 2000 --log_interval 50 --resume_checkpoint ./pretrain_model//model310000.pt --gpu 7 "
SIAMESE_FLAGS="--siamese_path ./OCR_siamese_utils/output/targetlist_lr0.01/bit.pth.tar"
OCR_FLAGS="--ocr_path ./OCR_siamese_utils/demo_downgrade.pth.tar"
DATASET_PATH="dataset/BOA" #change to point to your dataset path 
CHECKPOINT_PATH="saved/logo_BOA"
OPENAI_LOGDIR=$CHECKPOINT_PATH python scripts/image_train.py --data_dir $DATASET_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $SIAMESE_FLAGS $OCR_FLAGS
