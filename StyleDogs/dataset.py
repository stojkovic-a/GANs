import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
from PIL import Image
import struct
from array import array
import config as conf
from glob import glob

class RealDataset(Dataset):
    def __init__(self,image_path,transforms=None,image_size=conf.image_size) -> None:
        self.image_size=image_size
        self.transforms=transforms
        self.image_path=image_path
        self.image_files=sorted(glob(os.path.join(self.image_path,'**','*.jpg'),recursive=True))
        assert len(self.image_files)>0
        print(len(self.image_files))
        # print(self.image_files[10])
        # with open(self.image_path,'rb') as f:
        #     magic,size,rows,cols=struct.unpack(">IIII",f.read(16))
        #     assert rows==cols
        #     self.size=size
        #     if magic!=2051:
        #         raise ValueError(f'Magic number mismatch, expected 2051 got {magic} :/')
        #     image_data=array("B",f.read())

        # self.images=[]
        # for i in range(int(self.start_percent*self.size),int(min(1,(self.start_percent+self.count_percent))*self.size),1):
        #     self.images.append([0]*rows*cols)
        # for i in range(0, int(min(1,self.count_percent)*self.size),1):
        #     index=i+int(self.start_percent*self.size)
        #     if index>=self.size:
        #         raise IndexError("index error")
        #     img=np.array(image_data[index*rows*cols:(index+1)*rows*cols])
        #     img=img.reshape(rows,cols)
        #     delta=int((self.image_size-rows)/2)
        #     assert delta>=0
        #     img=np.pad(img,delta,mode='constant',constant_values=0)
        #     self.images[i][:]=img
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,idx):
        # img=np.array(Image.open(self.image_files[idx]).convert('RGB'))
        img=Image.open(self.image_files[idx]).convert('RGB')
        # if img.shape[2]==4:
        #     print()
        #     print()
        #     print()
        #     print(self.image_files[idx])
        #     print()
        #     print()
        #     print()
        # if img.shape[0]>img.shape[1]:
        #     print()
        #     print()
        #     print()
        #     print('dimesions cudne',self.image_files[idx])
        #     print()
        #     print()
        #     print()
        if self.transforms:
            img=self.transforms(img)
        return img

    
    
# if __name__=='__main__':
#     d=RealDataset(conf.image_dir,None,conf.image_size)
        