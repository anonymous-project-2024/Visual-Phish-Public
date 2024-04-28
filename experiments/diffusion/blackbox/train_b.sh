MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 2 --save_interval 2000 --log_interval 50 --resume_checkpoint ./deep_learning/guided-diffusion-sxela/logo_expand/model310000.pt --gpu 7 "
#surrogate
SIAMESE_FLAGS="--siamese_path ./phishpedia_siamese/resnetv2_rgb_new.pth.tar" 
DATASET_PATH="dataset/BOA" #change to point to your dataset path 
CHECKPOINT_PATH="saved/logo_BOA_b"
OPENAI_LOGDIR=$CHECKPOINT_PATH python scripts/image_train_mr.py --data_dir $DATASET_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $SIAMESE_FLAGS 