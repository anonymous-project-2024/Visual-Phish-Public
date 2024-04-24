# https://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
import os

def gen_image(arr):

    fig = np.around((arr) * 255.0)
    # fig = (arr + 0.5) * 255
    # fig = fig.astype(np.uint8).squeeze()
    fig = fig.astype(np.uint8).squeeze()
    img = Image.fromarray(fig)

    return img



# ATT6 rotate by 7 degree

# rotate_files = ["Apple_4.png", "Apple_11.png", "Apple_13.png", "Apple_16.png", "Apple_18.png", "Apple_22.png"]
# for rf in rotate_files:
#     orig_img = Image.open("{}".format(rf))
#     print("img size ", orig_img.size)
#     rotater = T.RandomRotation(degrees=(1, 2)) # rotate by 2 degree
#     rotated_imgs = rotater(orig_img) 
#     img_array = np.array(rotated_imgs) / 255.0
#     h,w = img_array.shape[0], img_array.shape[1]
#     new_img_array = img_array[int(h*0.02):int(w-w*0.015), int(h*0.02):int(w-w*0.015), :]
#     new_img = gen_image(new_img_array)
#     new_img.save("./rotate/rotated_{}_cropped.png".format(rf.split(".")[0].split("_")[-1]))

all_files =[i for i in os.listdir("./") if i.endswith("png")]
for rf in all_files:
    orig_img = Image.open("{}".format(rf))
    print("img size ", orig_img.size)
    rotater = T.RandomRotation(degrees=(3, 4)) # rotate by 2 degree
    rotated_imgs = rotater(orig_img) 
    img_array = np.array(rotated_imgs) / 255.0
    h,w = img_array.shape[0], img_array.shape[1]
    new_img_array = img_array[int(h*0.05):int(w-w*0.05), int(h*0.05):int(w-w*0.05), :]
    new_img = gen_image(new_img_array)
    new_img.save("./rotate/rotated_{}_cropped.png".format(rf.split(".")[0].split("_")[-1]))
# ATT1: rorate by 1 degree, 2 is too much
# orig_img = Image.open("ATT_1.png")

# rotater = T.RandomRotation(degrees=(1, 2)) # rotate by 2 degree
# rotated_imgs = rotater(orig_img) 
# img_array = np.array(rotated_imgs) / 255.0
# print("img array ", img_array.shape)
# new_img_array = img_array[2:100, 2:100, :]
# # print("new img array shape ", new_img_array.shape)
# # print("img array shape ", img_array.shape)
# new_img = gen_image(new_img_array)

# rotated_imgs.save("rotated_0.png")
# new_img.save("rotated_1_cropped.png")

# ATT 18
# a = [0,1,6,11,18,19]
# for i in (a):
#     path = "ATT_{}.png".format(i)
#     if os.path.exists(path):
#         orig_img = Image.open(path)
#     else:
#         orig_img = Image.open("ATT_18.jpg")
# all_files =[i for i in os.listdir("./") if i.endswith("png")]
# for i in all_files:
#     orig_img = Image.open(i)
#     augmenter = T.AugMix()
#     augmenter_imgs = [augmenter(orig_img) for _ in range(1)]

#     img_array = np.array(augmenter_imgs[0]) / 255.0
#     print("img array ", img_array.shape)
#     # new_img_array = img_array[6:366, 6:366, :]
#     # print("new img array shape ", new_img_array.shape)
#     # print("img array shape ", img_array.shape)
#     new_img = gen_image(img_array)

#     # rotated_imgs.save("rotated_.png")
#     new_img.save("./augmix/augmenter_{}.png".format(i.split(".")[0].split("_")[-1]))




