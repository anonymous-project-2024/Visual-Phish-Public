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
    ocr_model,
    ):
        super().__init__()
        self.siamese_model = model
        self.ocr_model = ocr_model

    def l2_norm(self, x):
        if len(x.shape):
            x = x.reshape((x.shape[0],-1))
        return F.normalize(x, p=2, dim=1)


    def pred_siamese_OCR(self, img, model, ocr_model, imshow=False, title=None, grayscale=False):
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
        
        ocr_emb = ocr_main2(image_path=img, model=ocr_model, height=None, width=None)
        # print("ocr emb out ", ocr_emb.shape)
        # ocr_emb = ocr_emb[0]

        ocr_emb = ocr_emb.to(device) 
        # print("ocr emb gradient? ", ocr_emb)
        # print("ocr emb shape device ", ocr_emb.shape)
        img = img_transforms(img)
        img = img.to(device)

        logo_feat = model.features3(img, ocr_emb)

        logo_feat = self.l2_norm(logo_feat)
        # print("logo feat gradient? shape", logo_feat)
        # logo_feat = logo_feat.view(-1)
        # print("logo feat gradient? shape ", logo_feat.shape)
        
    
        return logo_feat

    def siamese_loss(self, x1, x2):
        # print("start in siamese x1 grad_fn ", x1)
        # print("start in siamese x2 grad_fn ", x2)
        img_feat1 = self.pred_siamese_OCR(x1, self.siamese_model, self.ocr_model)
        
        img_feat2 = self.pred_siamese_OCR(x2, self.siamese_model, self.ocr_model)
       
  
        sim_list2 = []
        for i in range(img_feat1.shape[0]):
            sim_list2.append(img_feat1[i] @ img_feat2[i].T)
        
        sim_list2 = torch.stack(sim_list2)
     

        return sim_list2



# # to prove for the threshold
# def main():

#     def phishpedia_config_OCR_easy2(num_classes: int, weights_path: str, ocr_weights_path: str):
#         # load OCR model

#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#         ocr_model = ocr_model_config(checkpoint=ocr_weights_path)


#         #from .phishpedia_siamese.siamese_retrain.bit_pytorch.models import KNOWN_MODELS

#         from OCR_siamese_utils.siamese_unified.bit_pytorch.models import KNOWN_MODELS
#         model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

#         # Load weights
#         weights = torch.load(weights_path, map_location=device)
#         weights = weights['model'] if 'model' in weights.keys() else weights
#         new_state_dict = OrderedDict()
#         for k, v in weights.items():
#             if k.startswith('module'):
#                 name = k.split('module.')[1]
#             else:
#                 name = k
#             new_state_dict[name] = v

#         # for k, v in weights.items():
#         #     name = k.split('module.')[1]
#         #     new_state_dict[name] = v
#         model.load_state_dict(new_state_dict)
#         model.to(device)
#         model.eval()


#         return model, ocr_model



#     siamese_path="./OCR_siamese_utils/output/targetlist_lr0.01/bit.pth.tar"
#     ocr_path ="./OCR_siamese_utils/demo_downgrade.pth.tar"
#     transform = transforms.Compose([transforms.ToTensor()])

#     SIAMESE_MODEL, OCR_MODEL = phishpedia_config_OCR_easy2(num_classes=277,
#                     weights_path=os.path.join(os.path.dirname(__file__), siamese_path.replace('/', os.sep)), 
#                     ocr_weights_path=os.path.join(os.path.dirname(__file__), ocr_path.replace('/', os.sep))

#                     )
#     siamese_model =  SiameseModel(SIAMESE_MODEL, OCR_MODEL)
#     # x1 = Image.open("../deep_learning/expand_small_ps_correct/BOA/BOA_9.png")
#     # x1 = transform(x1)
#     # x1 = x1[None, :]
#     similarities = []
#     #folders = [folder for folder in os.listdir("../deep_learning/expand_small_ps_correct/")]
#     folders = ["Amazon2"]
#     for folder in folders:
#         path = os.listdir("../deep_learning/expand_small_ps_correct/{}".format(folder))[0]
#         x1 = Image.open("../deep_learning/expand_small_ps_correct/{}/{}".format(folder, path))
#         x1 = transform(x1)
#         x1 = x1[None, :]
#         for folder2 in folders:
#             print("folder {} compares {}".format(folder, folder2))
#             path = os.listdir("../deep_learning/expand_small_ps_correct/{}".format(folder2))[0]
#             x2 = Image.open("../deep_learning/expand_small_ps_correct/{}/{}".format(folder2, path))
#             x2 = transform(x2)
#             x2 = x2[None, :]
            
#             similarity = siamese_model.siamese_loss(x1, x2)
#             similarities.append(similarity)
#             print("similarity ", similarity)
#     print("similarities ", similarities)

# if __name__ == "__main__":
#     main()