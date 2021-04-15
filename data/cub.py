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


class CUB(Dataset):

    def __init__(self, root=MyPath.db_root_dir('cub'), train=True, transform=None):

        super(CUB, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train
        self.imgNames = []
        self.classes = []

        with open(root + 'classes.txt', 'r') as classesFile:
            for line in classesFile.readline():
                print(line)
                print(line.split(' ')[1])
                self.classes.append(line.split(' ')[1].split('.')[1])

        isTrainList = []
        with open(root + 'train_test_split.txt', 'r') as splitFile:
            for line in splitFile.readlines():
                isTrainList.append(int(line.strip().split(' ')[1]))

        classLabels = []
        with open(root + 'image_class_labels.txt', 'r') as labelsFile:
            for line in labelsFile.readlines():
                classLabels.append(int(line.strip().split(' ')[1]) - 1)

        imagePaths = []
        with open(root + 'images.txt', 'r') as imagesFile:
            for line in imagesFile.readlines():
                imagePaths.append(root + 'images/' + line.strip().split(' ')[1])

        self.imagePaths = []
        self.classLabels = []
        for i, isTrain in enumerate(isTrainList):
            if isTrain == train:
                self.imagePaths.append(imagePaths[i])
                self.classLabels.append(classLabels[i])


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img = Image.open(self.imagePaths[index]).convert("RGB")
        target = self.classLabels[index]   
        imgSize = img.size#(img.shape[0], img.shape[1])    

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': imgSize, 'index': index, 'class_name': self.imagePaths[index].split('/')[-2].split('.')[1]}}
        
        return out

    def get_image(self, index):
        img = Image.open(self.imagePaths[index]).convert("RGB")
        return img
        
    def __len__(self):
        return len(self.imagePaths)
