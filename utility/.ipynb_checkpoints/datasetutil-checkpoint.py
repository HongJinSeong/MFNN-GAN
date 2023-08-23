import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import glob as _glob
import csv
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import skimage.io as iio
import skimage.color  as icol
import skimage.transform as skiT
import utility.dtype as dtype
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
import pandas as pd
from math import log, cos, pi, floor
import math


csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 ) #MSE 구하는 코드

    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) #PSNR구하는 코드


#region 데이터 저장용

def writecsv(csvname,contents):
    f = open(csvname, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(contents)
    f.close()

def imwrite(image, path,gray,resize, **plugin_args):
    #normalize -1 to 1
    mins=np.min(image,axis=(0,1))
    maxs=np.max(image,axis=(0,1))
    for i in range(image.shape[2]):
        if mins[i]<-1 or maxs[i]>1:
            image[:,:,i]=(((image[:,:,i]-mins[i])/(maxs[i]-mins[i]))*2)-1
    """Save a [-1.0, 1.0] image."""
    # 1 channel 로 저장하고 싶다면 해당 파라미터 True
    if gray==True:
        image=icol.rgb2gray(image)
    if resize==True:
        image=skiT.resize(image,(70,180))
    iio.imsave(path, dtype.im2uint(image), **plugin_args)


def mkdir(paths):
  if not isinstance(paths, (list, tuple)):
    paths = [paths]
  for path in paths:
    if not os.path.exists(path):
      os.makedirs(path)

def split(path):
  """Return dir, name, ext."""
  dir, name_ext = os.path.split(path)
  name, ext = os.path.splitext(name_ext)
  return dir, name, ext


def glob(dir, pats, recursive=False):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches
#endregion

def split(path):
  """Return dir, name, ext."""
  dir, name_ext = os.path.split(path)
  name, ext = os.path.splitext(name_ext)
  return dir, name, ext

def cosine_decay(step,alpha,decay_steps):
  step = min(step, decay_steps)
  cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  return decayed

#region pytorch Custom dataset
class FingerveinDataset(Dataset):
    def __init__(self, blurlist, clearlist, transform1=None, transform2=None, csv=''):
        self.blurlist = blurlist
        self.clearlist = clearlist
        self.transform1 = transform1
        self.transform2 = transform2
        self.label_info = pd.read_csv(csv)

    def __len__(self):
        return len(self.blurlist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        blur_name = self.blurlist[idx]
        clear_name = self.clearlist[idx]

        target_info = self.label_info[self.label_info['ls'].str.contains(str.replace(blur_name, '\\', '/').split('/')[-1][1:])]


        if target_info.__len__() > 0:
            ## gaussian noise and 어둡게
            if target_info['col'].values[0] == 'B':
                lbl = np.array([0, 1, 0])
            ## only 밝게
            else:
                lbl = np.array([0, 0, 1])
        ## only gaussian noise
        else:
            lbl = np.array([1, 0, 0])

        img1 = iio.imread(blur_name)
        img2 = iio.imread(clear_name)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        params = transforms.RandomResizedCrop.get_params(img1, scale=(0.7, 1.0), ratio=(0.75, 1.333))

        ##input shape 확인하고 resize 필요
        img1 = TF.crop(img1, *params)
        img1=img1.resize((224, 224))
        img2 = TF.crop(img2, *params)
        img2 = img2.resize((224, 224))

        ## 1 channel 이미지를 input으로 받기 때문에 3 channel 로 변경

        img1 = (self.transform1(img1)*2)-1 # PIL Image로 바꾸면 value 가 0~1로 정규화 되기 때문에 img2와 맞추기 위해 -1~1의 값으로 조정
        img1 = torch.cat((img1,img1,img1),dim=0)
        img2 = (self.transform2(img2)*2)-1 # PIL Image로 바꾸면 value 가 0~1로 정규화 되기 때문에 img2와 맞추기 위해 -1~1의 값으로 조정
        img2 = torch.cat((img2, img2, img2), dim=0)
        return img1,img2,lbl

class FingerveinDataset_test(Dataset):
    def __init__(self,blurlist,originlist,transform=None):
        self.blurlist=blurlist
        self.originlist=originlist
        self.transform=transform

    def __len__(self):
        return len(self.blurlist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        blur_name=self.blurlist[idx]
        origin_name=self.originlist[idx]


        #split 데이터 확인해보고 인덱스에 따라서 폴더와 파일명 필요 예상으로는 2 폴더 3 파일명
        blur_name_split=blur_name.replace('\\','/').split('/')

        img1 = cv2.imread(blur_name,cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(origin_name,cv2.IMREAD_GRAYSCALE)

        #img1 = iio.imread(blur_name)
        #img2 = iio.imread(origin_name)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        img1 = img1.resize((224, 224))
        img2 = img2.resize((224, 224))

        img1 = (self.transform(img1) * 2) - 1  # PIL Image로 바꾸면 value 가 0~1로 정규화 되기 때문에 img2와 맞추기 위해 -1~1의 값으로 조정
        img1 = torch.cat((img1, img1, img1), dim=0)
        img2 = (self.transform(img2) * 2) - 1  # PIL Image로 바꾸면 value 가 0~1로 정규화 되기 때문에 img2와 맞추기 위해 -1~1의 값으로 조정
        img2 = torch.cat((img2, img2, img2), dim=0)

        return blur_name_split[2],blur_name_split[3],img1,img2


#endregion