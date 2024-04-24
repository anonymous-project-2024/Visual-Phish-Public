"""
Train a diffusion model on images.
"""

import argparse, os

from guided_diffusion import logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch
# device = 'cuda'

# torch.distributed.init_process_group('nccl', init_method='', world_size=1, rank=0)


def main():
    args = create_argparser().parse_args()
   
    if args.logdir != '':
        os.environ['OPENAI_LOGDIR'] = args.logdir
        print('set output to ',os.environ['OPENAI_LOGDIR'])
    logger.configure()

    logger.log("creating model and diffusion...")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("using gpu ", str(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, diffusion, siamese_model = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(device)
    siamese_model.siamese_model.to(device)
    siamese_model.ocr_model.to(device)
    # model = torch.nn.paralle.DistributedDataParallel(model.to(device))
    # siamese_model = torch.nn.paralle.DistributedDataParallel(siamese_model.siamese_model.to(device))
    # siamese_model.ocr_model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) # put differents weights on the sampling process

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
   
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        siamese_model=siamese_model,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        start_scaling=args.start_scaling, 
        gpu = args.gpu
        # mask_path = args.mask_path
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_dir="",
        # mask_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        logdir = '',
        start_scaling=0,
        gpu=0
        # siamese_path=''
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
