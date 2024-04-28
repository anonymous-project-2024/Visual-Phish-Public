# import os
# import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
# import matplotlib.pyplot as plt
# from torch import nn
import torch.nn.functional as F
# import torch
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from torch.backends import cudnn
# from tqdm import tqdm
# from .phishpedia_siamese.inference import siamese_inference, pred_siamese
# from .OCR_siamese_utils.inference import siamese_inference_OCR, pred_siamese_OCR
# from .OCR_siamese_utils.demo import ocr_model_config
# import yaml
# import subprocess
from .OCR_siamese_utils.demo import ocr_main, ocr_main2
# from skimage.io import imread
# from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torch
import os
from .OCR_siamese_utils.demo import ocr_model_config
from collections import OrderedDict

class SiameseModel():
    def __init__(
    self,
    model,
  
    ):
        super().__init__()
        self.siamese_model = model

    def l2_norm(self, x):
        if len(x.shape):
            x = x.reshape((x.shape[0],-1))
        return F.normalize(x, p=2, dim=1)


    def pred_siamese_OCR(self, img, model, imshow=False, title=None, grayscale=False):
        '''
        Inference for a single image with OCR enhanced model
        :param img_path: image path in str or image in PIL.Image
        :param model: Siamese model to make inference
        :param ocr_model: pretrained OCR model
        :param imshow: enable display of image or not
        :param title: title of displayed image
        :param grayscale: convert image to grayscale or not
        :return feature embedding of shape (2048,)
        '''

        for params in model.parameters():
                params.requires_grad = False

        img_size = 224
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print("what is device ", device)
        img_transforms = transforms.Compose(
            [
            transforms.Normalize(mean=mean, std=std),
            ])
        
        img = F.interpolate(img, size=img_size, mode='bicubic') 
        
        # print("ocr emb gradient? ", ocr_emb)
        # print("ocr emb shape device ", ocr_emb.shape)
        img = img_transforms(img)
        img = img.to(device)

        logo_feat = model.features(img)
        logo_feat = self.l2_norm(logo_feat)
    
        return logo_feat

    def siamese_loss(self, x1, x2):
        # print("start in siamese x1 grad_fn ", x1)
        # print("start in siamese x2 grad_fn ", x2)
        img_feat1 = self.pred_siamese_OCR(x1, self.siamese_model)
        
        img_feat2 = self.pred_siamese_OCR(x2, self.siamese_model)
       
  
        sim_list2 = []
        for i in range(img_feat1.shape[0]):
            sim_list2.append(img_feat1[i] @ img_feat2[i].T)
        
        sim_list2 = torch.stack(sim_list2)
     

        return sim_list2
