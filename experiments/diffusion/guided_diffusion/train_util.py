import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import torch
from torch.autograd import Variable
# 
from PIL import Image
import numpy as np

# make_dot was moved to https://github.com/szagoruyko/pytorchviz
# from torchviz import make_dot

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

# device = 'cuda'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# # device = 'cuda'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        siamese_model,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        start_scaling=0.1,
        gpu,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        print("self.microbatch ", self.microbatch)
        self.siamese_model = siamese_model
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size 
        self.start_scaling = start_scaling
        # self.mask_path = mask_path
        print("start scaling is!!!!", self.start_scaling)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        global device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
      
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = False
            self.ddp_model = self.model
    
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(th.load(
                        resume_checkpoint, map_location='cuda')
                )



    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if True:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(
                    ema_checkpoint, map_location=device
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)


        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location=device
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # print("what is lr_anneal_steps ", self.lr_anneal_steps)
            # print("what is step ", self.step)
            torch.backends.cudnn.enabled=False
            batch, cond, path = next(self.data)
            # print("path ", path)
            # print("next batch called ", batch)  

            ####=====================================
            # if self.step < 150000:
            #     self.sc = self.start_scaling
            #     # print("scaling factor ", self.sc)
            #     self.run_step(batch, cond)
            # elif self.step < 200000: 
            #     self.sc = self.start_scaling + 0.0015
            #     # print("scaling factor ",self.sc)
            #     self.run_step(batch, cond)
            # elif self.step < 250000:
            #     self.sc = float((self.start_scaling + 0.0015) / 0.75)
            #     # print("scaling factor ", self.sc)
            #     self.run_step(batch, cond)
            # else:
            #     self.sc = float((self.start_scaling + 0.0015) / 0.15)
            #     # print("scaling factor ", self.sc)
            #     self.run_step(batch, cond)
            ###==========================================
            # if self.step < 20000:
            #     self.sc = float(0.001)  # 0.005 -> 0.002
            #     self.run_step(batch, cond, path)

            # elif self.step < 40000:
            #     self.sc = float(0.002)  # 0.005 -> 0.002
            #     self.run_step(batch, cond, path)
            # elif self.step < 60000:
            #     self.sc = float(0.003)  # 0.005 -> 0.002  # 370000
            #     self.run_step(batch, cond, path)

            # elif self.step < 80000:
            #     self.sc = float(0.004)  # 0.005 -> 0.002 # 390000
            #     self.run_step(batch, cond, path)
            # elif self.step < 100000:
            #     self.sc = float(0.005)  # 0.005 -> 0.002  # 410000
            #     self.run_step(batch, cond, path)


            if self.step < 20000:
                self.sc = float(0.002)  # 0.005 -> 0.002
                self.run_step(batch, cond, path)

            elif self.step < 40000:
                self.sc = float(0.004)  # 0.005 -> 0.002
                self.run_step(batch, cond, path)
            elif self.step < 60000:
                self.sc = float(0.006)  # 0.005 -> 0.002  # 370000
                self.run_step(batch, cond, path)

            elif self.step < 80000:
                self.sc = float(0.008)  # 0.005 -> 0.002 # 390000
                self.run_step(batch, cond, path)  
            else:
                self.sc = float(0.01)  # 0.005 -> 0.002
                self.run_step(batch, cond, path)
            # elif self.step < 40000:
            #     self.sc = float((self.start_scaling + 0.0015) / 0.375)  # 0.004
            #     self.run_step(batch, cond, path)
            # else:
            #     self.sc = float((self.start_scaling + 0.0015) / 0.3) # 0.005
            #     self.run_step(batch, cond, path)

            #float((self.start_scaling + 0.0015) / 0.1) # 0.015
            # 0.02
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                print("scaling factor ", self.sc)
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step % 1000 == 0:
                print("step is ", self.step)
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, path):
        self.forward_backward(batch, cond, path)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, path):
        self.mp_trainer.zero_grad()  # unet
        # print("update before for ddp model ============")
        # for name, parms in self.ddp_model.named_parameters():
        #     print("-->name: ", name)
        #     print("-->para: ", parms)
        #     print("-->grad requires ", parms.requires_grad)
        #     print("-->grad value ", parms.grad)
            
        # print("siamese params after ============")
        # for parms in self.siamese_model.siamese_model.parameters():
        #     print("-->para ", parms)
        #     print("grad requiires ", parms.requires_grad)
        #     print("-->grad value ", parms.grad)
        
        # #make_dot(losses["loss"], params=dict(list(self.ddp_model.named_parameters())), show_attrs=True, show_saved=True).render("torchviz", format="png")
    
        # print("ocr params after ============")
        # for parms in self.siamese_model.ocr_model.parameters():
        #     print("-->para ", parms)
        #     print("grad requiires ", parms.requires_grad)
        #     print("-->grad value ", parms.grad)

        for i in range(0, batch.shape[0], self.microbatch):
            # print("should be i increment", i)
            # print("batch shape", batch.shape)
            # print("microbatch ", self.microbatch)
            micro = batch[i : i + self.microbatch].to(device)
            
      
            xt_sample = ((micro+ 1) * 127.5).clamp(0, 255).to(th.uint8)
  
            xt_sample = xt_sample.permute(0, 2, 3, 1)
            xt_img =  Image.fromarray(xt_sample.cpu().numpy()[0])
            # print("noise does it have gradients", noise)x
            # xt_img.save("./xt/micro_{}.png".format(self.step))


            micro_cond = {
                k: v[i : i + self.microbatch].to(device)
                for k, v in cond.items()
            }
          
            paths = path['path']
            # mask_imgs = []
            # print("type of micro ", type(micro))
            width, height = micro.shape[2], micro.shape[3]
            # print("width and height ", width, height)
            # for path in paths:
            #     mask_path = "{}/mask_{}".format(self.mask_path, path.split("/")[-1])
            #     mask_img = Image.open(mask_path)
            #     mask_img = mask_img.convert('RGB')
            #     mask_img = mask_img.resize((width, height))
            #     mask_img = (np.array(mask_img)/ 255.0).astype(np.float64)  # 0 and 1
            #     mask_img = np.transpose(mask_img, [2, 0, 1])
            #     # print("mask img shape ", mask_img.shape)
            #     mask_imgs.append(mask_img)
            # mask_imgs = np.array(mask_imgs)
            # mask_imgs = torch.FloatTensor(mask_imgs)
            # mask_imgs = mask_imgs.to(device)
      
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], device)
            # print("what is t is essential!!!!===== ", t)
            # print("t shape ", t.shape)
            # print("weights ", weights)


            # except for micro, also need to pass in logo_feat
            compute_losses = functools.partial(
                self.diffusion.training_losses,  # it is just call the function
                self.ddp_model,
                self.siamese_model,
                self.sc,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            # print("losses shape ", losses["loss"].shape)
            # print("losses items ", losses.items())
            # print("weights shape ", weights.shape)
            loss = (losses["loss"] * weights).mean()
             
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

            # print("update after for ddp model ============")
            # for name, parms in self.ddp_model.named_parameters():
            #     print("-->name: ", name)
            #     print("-->para: ", parms)
            #     print("-->grad requires ", parms.requires_grad)
            #     print("-->grad value ", parms.grad)
               
            # print("siamese params after ============")
            # for parms in self.siamese_model.siamese_model.parameters():
            #     print("-->para ", parms)
            #     print("grad requiires ", parms.requires_grad)
            #     print("-->grad value ", parms.grad)
           
            #make_dot(losses["loss"], params=dict(list(self.ddp_model.named_parameters())), show_attrs=True, show_saved=True).render("torchviz", format="png")
        
            # print("ocr params after ============")
            # for parms in self.siamese_model.ocr_model.parameters():
            #     print("-->para ", parms)
            #     print("grad requiires ", parms.requires_grad)
            #     print("-->grad value ", parms.grad)
     
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if True:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if True:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)