"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class PASCALVOC(Dataset):

    def __init__(self, root=MyPath.db_root_dir('pascal-voc'), train=True, transform=None):

        super(PASCALVOC, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train
        self.imgNames = []
        self.classNames = []
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.classDict = {}
        for i, name in enumerate(self.classes):
            self.classDict[name] = i# + 1

        with open(self.root + '/bndBoxImageLabels.txt', 'r') as labelFile:
            for line in labelFile.readlines():
                self.imgNames.append(line.split(' ')[0])
                self.classNames.append(line.split(' ')[1].strip())


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        imgFileName = os.path.join(str(self.root) + '/BBoxImages/' + str(self.imgNames[index]) + '.jpg')
        img = Image.open(imgFileName).convert("RGB")
        className = self.classNames[index]
        target = self.classDict[className]    
        imgSize = img.size#(img.shape[0], img.shape[1])    

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': imgSize, 'index': index, 'class_name': className}}
        
        return out

    def get_image(self, index):
        imgFileName = os.path.join(str(self.root) + '/BBoxImages/' + str(self.imgNames[index]))
        img = Image.open(imgFileName).convert("RGB")
        return img
        
    def __len__(self):
        return len(self.imgNames)
