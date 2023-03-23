import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import copy
import os
import numpy as np
import nibabel as nib
from config import DatasetParams
from pathlib import Path

#Function to find the AC coordinates from text file
def findAC(myfile):
    myfile = open(myfile,'rt')
    
    i = 1
    myprev = ""
    for myline in myfile:              # For each line, read to a string,
        if "AC (i,j,k)" in myprev:
            mylist = myline.split(maxsplit=-1)
            ac = [round(float(c)) for c in mylist]
            return ac
        myprev = myline
        i += 1

#Function to find the PC coordinates from text file
def findPC(myfile):
    myfile = open(myfile,'rt')
    
    i = 1
    myprev = ""
    for myline in myfile:              # For each line, read to a string,
        if "PC (i,j,k)" in myprev:
            mylist = myline.split(maxsplit=-1)
            return [round(float(i)) for i in mylist]
        myprev = myline
        i=i+1
    

#Function to create Heatmap     
def createHeatmap(input_image, coord):
    
    #Initialising the Binary Mask
    maskimage = np.zeros(input_image.shape)
    
    #Setting up the Radius of the AC PC indicators
    radius = 2

    #Getting the AC and PC Coordinates
    ac = findAC(coord)
    pc = findPC(coord)
    
    #Creating the AC indicator
    for x in range(ac[0]-radius, ac[0]+radius+1):
        for y in range(ac[1]-radius, ac[1]+radius+1):
            for z in range(ac[2]-radius, ac[2]+radius+1):  
                deb = radius - abs(ac[0]-x) - abs(ac[1]-y) - abs(ac[2]-z) 
                if (deb)>=0:
                    maskimage[x,y,z] = 1
    
    #Creating the PC indicator
    for x in range(pc[0]-radius, pc[0]+radius+1):
        for y in range(pc[1]-radius, pc[1]+radius+1):
            for z in range(pc[2]-radius, pc[2]+radius+1): 
                deb = radius - abs(pc[0]-x) - abs(pc[1]-y) - abs(pc[2]-z) 
                if (deb)>=0: 
                    maskimage[x,y,z] = 1
    
    return maskimage    

class OASISDataset(Dataset):
    
    #Initialising function
    def __init__(self, dir_path, split_ratios = [0.8, 0.1, 0.1], transforms = None, mode = None) -> None:
        self.splits = split_ratios
        self.transform = transforms
        
        # train_val_test = [(x-1) for x in train_val_test]
        self.mode = mode

        #Getting the image and the Label file from the directory
        image_dir = Path(dir_path)
        self.image_path_list = list(image_dir.glob("*.nii"))
        #Calculating split number of images
        num_training_imgs =  len(self.image_path_list)
        train_val_test = [int(x * num_training_imgs) for x in split_ratios]
        if(sum(train_val_test) != num_training_imgs): train_val_test[0] += (num_training_imgs - sum(train_val_test))
        train_val_test = [x for x in train_val_test if x!=0]
        
        #Spliting dataset
        samples = self.image_path_list
        shuffle(samples)
        self.train = samples[0:train_val_test[0]]
        self.val = samples[train_val_test[0]:train_val_test[0] + train_val_test[1]]
        self.test = samples[train_val_test[1]:train_val_test[1] + train_val_test[2]]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.train)
        elif self.mode == "val":
            return len(self.val)
        elif self.mode == "test":
            return len(self.test)    
    
    #Function to get the data items 
    def __getitem__(self,idx):
        
        image_object = nib.load(self.image_path_list[idx]).get_fdata()
        inter1 = str(self.image_path_list[idx])
        inter2 = inter1.split(".nii")
        #print(self.inter2)
        label_path = Path(inter2[0]+"_ACPC.txt")

        label_object = createHeatmap(image_object, label_path)
        
        image_array = np.moveaxis(image_object, -1, 0)
        label_array = np.moveaxis(label_object, -1, 0)
        
        processed_out = {'image': image_array, 'label': label_array}

        if self.transform:
            if self.mode == "train":
                processed_out = self.transform[0](processed_out)
            elif self.mode == "val":
                processed_out = self.transform[1](processed_out)
            elif self.mode == "test":
                processed_out = self.transform[2](processed_out)
        
        #The output numpy array is in channel-first format
        return processed_out
    
def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    The utility function to generate splitted train, validation and test dataloaders
    
    Note: all the configs to generate dataloaders in included in "config.py"
    """
    dataset = OASISDataset(dir_path=DatasetParams.DATASET_PATH, transforms=[train_transforms, val_transforms, test_transforms])

    #Spliting dataset and building their respective DataLoaders
    train_set, val_set, test_set = copy.deepcopy(dataset), copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_set.set_mode('train')
    val_set.set_mode('val')
    test_set.set_mode('test')
    train_dataloader = DataLoader(dataset= train_set, batch_size= DatasetParams.TRAIN_BATCH_SIZE, shuffle= False)
    val_dataloader = DataLoader(dataset= val_set, batch_size= DatasetParams.VAL_BATCH_SIZE, shuffle= False)
    test_dataloader = DataLoader(dataset= test_set, batch_size= DatasetParams.TEST_BATCH_SIZE, shuffle= False)
    
    return train_dataloader, val_dataloader, test_dataloader