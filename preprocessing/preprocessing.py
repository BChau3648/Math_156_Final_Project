import os
from tqdm import tqdm
import pickle

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
from torchvision.io import read_image 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

def calculate_channel_mean():
    # calculating mean over each RGB channel
    means_sum = torch.zeros(3)
    num_img = 0
    # iterate over each class folder inside dataset
    for land_class in os.listdir(data_path):
        print(land_class)
        land_class_path = os.path.join(data_path, land_class)
        # iterate over each image for each class
        for file in tqdm(os.listdir(land_class_path)):
            img_path = os.path.join(land_class_path, file)
            img = read_image(img_path).to(torch.float64) 
            means_sum += torch.mean(img, dim=(1,2))
            num_img += 1
        print('==========================================')
    channel_mean = means_sum / num_img
    return channel_mean, num_img

# calculates average of squared deviations across channels
def sample_var(x, channel_mean):
    return torch.mean((x - channel_mean[:, None, None])**2, dim = (1,2))

def calculate_channel_sd(channel_mean, num_img):
    # calculating variance over each RGB channel
    vars_sum = torch.zeros(3)
    # iterate over each class folder inside dataset
    for land_class in os.listdir(data_path):
        print(land_class)
        land_class_path = os.path.join(data_path, land_class)
        # iterate over each image for each class
        for file in tqdm(os.listdir(land_class_path)):
            img_path = os.path.join(land_class_path, file)
            img = read_image(img_path).to(torch.float64) 
            vars_sum += sample_var(img, channel_mean)
        print('==========================================')
    # take square root to get standard deviation
    channel_sd = torch.sqrt(vars_sum / num_img)
    return(channel_sd)

class EuroSATDataset(Dataset):
    def __init__(self, data_path, preprocessing_stats_path, transform=False):
        self.data_path = data_path
        # getting preprocessing statistics
        with open(preprocessing_stats_path, 'rb') as f:
                preprocessing_stats = pickle.load(f)
        channel_mean = preprocessing_stats['mean']
        channel_sd = preprocessing_stats['sd']
        # setting normalization and augmentation
        self.transform = transform
        if self.transform:
            self.transform = v2.Compose([
                v2.Normalize(mean=channel_mean, std=channel_sd)
            ])
            # TODO: check to see if these are necessary
            # self.transform = v2.Compose([
            #     v2.Normalize(mean=channel_mean, std=channel_sd),
            #     v2.RandomHorizontalFlip(p=0.5),
            #     v2.RandomVerticalFlip(p=0.5),
            #     v2.GaussianBlur((3,3), (0.5,1))
            # ])

        self.sorted_class_names = sorted(os.listdir(self.data_path))
        self.num_classes = len(os.listdir(self.data_path))
        self.num_img_per_class = torch.zeros(self.num_classes, dtype=torch.int)

        # getting cumsum number of images per class sorted alphabetically        
        for i, land_class in enumerate(self.sorted_class_names):
            self.num_img_per_class[i] = len(os.listdir(os.path.join(self.data_path, land_class)))
        self.cumsum_img_per_class = torch.cumsum(self.num_img_per_class, dim=0)
        
    def __len__(self):
        return torch.sum(self.num_img_per_class)

    def __getitem__(self, idx):
        # calculating which class folder to read from
        idx_diff = self.cumsum_img_per_class - idx
        class_idx = torch.sum(idx_diff <= 0)
        # recalculating index if going to other folders
        if class_idx != 0:
            idx = idx - self.cumsum_img_per_class[class_idx - 1]
        
        # getting image tensor and class name
        class_name = self.sorted_class_names[class_idx]
        class_path = os.path.join(self.data_path, class_name)
        img_name = os.listdir(class_path)[idx]
        img_path = os.path.join(class_path, img_name)
        img = read_image(img_path).to(torch.float64)
    
        # preprocessing and augmenting data
        if self.transform:
            img = self.transform(img)
        # one-hot encoding label according to alphabetical order
        onehot_label = torch.nn.functional.one_hot(class_idx, num_classes=self.num_classes)
        
        sample = {'image': img, 'land_use': onehot_label}
        return sample

if __name__ == '__main__':
    # path to EuroSAT dataset
    data_path = './EuroSAT_RGB'
    # do not run again if preprocessing statistics already saved in preprocessing folder
    already_preprocessed = (os.path.isfile('./preprocessing/preprocessing_stats.p') or 
                            os.path.isfile('./preprocessing/preprocessing_stats.pkl'))
    
    if not already_preprocessed:
        channel_mean, num_img = calculate_channel_mean()
        channel_sd = calculate_channel_sd(channel_mean, num_img)
        
        preprocessing_stats = {
            'mean': channel_mean, 
            'sd': channel_sd, 
            'num_img': num_img
        }
        with open('./preprocessing/preprocessing_stats.pkl', 'wb') as f:
            pickle.dump(preprocessing_stats, f)
        eurosat = EuroSATDataset(data_path, './preprocessing/preprocessing_stats.pkl')
    else:
        try: 
            eurosat = EuroSATDataset(data_path, './preprocessing/preprocessing_stats.pkl')
        except: 
            eurosat = EuroSATDataset(data_path, './preprocessing/preprocessing_stats.p')

    # testing to see if dataset iterates correctly
    for i in range(0, 27000, 100):
        eurosat[i] 
        if i % 2500 == 0:
            print(f"{i}: {eurosat[i]['land_use']}")

    eurosat[26999]
    try:
        eurosat[27000]
    except:
        print('Out of bounds works')
    