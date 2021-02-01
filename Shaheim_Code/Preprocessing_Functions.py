
"""
Data preprocessing functions for MR Cardiac Thermometry magnitude images for
Pytorch module integration 

Starter Code: https://gist.github.com/somada141/8dd67a02e330a657cf9e

              https://stackoverflow.com/questions/33180258/
              optimize-performance-for-calculation-of-euclidean
              -distance-between-two-images

"""

import os
import numpy as np
import pydicom
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def Get_DICOM_FILES(pth):

    lstFilesDCM = []
    
    for dirName, subdirList, fileList in os.walk(pth):
        
        for filename in fileList:
            
            if ".dcm" in filename.lower():
                
                lstFilesDCM.append(os.path.join(dirName,filename))  
    
    Data_array = []
    
    for filenameDCM in lstFilesDCM:
        
        ds = pydicom.read_file(filenameDCM)
        arr = ds.pixel_array
        arr = arr - arr.mean()
        arr = arr / arr.max()
        Data_array.append(arr)
    
    return Data_array,lstFilesDCM


def L2_norm(i1, i2):
    
    return np.sum((i1-i2)**2)


def Find_base_pair(heated_image,base_images):
    
    lowest_dis = 1e10
    
    best_image_indx = 0
    
    for i,base_image in enumerate(base_images):
        
        distance = L2_norm(heated_image,base_image)
        
        if distance < lowest_dis:
            
            best_image_indx = i
            lowest_dis = distance
            
        
    #print("Lowest L2 norm: ", lowest_dis)
    
    return base_images[best_image_indx]
            

def Get_Base_pairs(heated_images,base_images):
    
    base_pairs = []
    
    for heated_image in heated_images:
        
        base_image = Find_base_pair(heated_image,base_images)
        base_pairs.append(base_image)
        
    return base_pairs


            
def Get_Loader(heated_images,base_images,batch_size=1,spatial_trans = False,shuffle_bool = True):
    
    base_pair_images = Get_Base_pairs(heated_images,base_images)
    
    in_images = []
    out_images = []
    
    for i in range(len(heated_images)):
        
        heat_image = heated_images[i]
        base_image = base_pair_images[i]
        in_image = heat_image[None,:,:]
        out_image = base_image[None,:,:]
        in_images.append(in_image)
        out_images.append(out_image)
    

        
    tensor_x = torch.Tensor(in_images) 
    tensor_y = torch.Tensor(out_images)


        
    dataset = TensorDataset(tensor_x,tensor_y)
    dataloader = DataLoader(dataset,shuffle=shuffle_bool)
    
    return dataloader

def Get_Loader_Valid(heated_images,base_images,batch_size=1):
    
    base_pair_images = Get_Base_pairs(heated_images,base_images)
    
    in_images = []
    out_images = []
    
    for i in range(len(heated_images)):
        
        heat_image = heated_images[i]
        base_image = base_pair_images[i]
        in_image = heat_image[None,:,:]
        out_image = base_image[None,:,:]
        in_images.append(in_image)
        out_images.append(out_image)
        
    in_train, in_valid, out_train, out_valid = train_test_split(in_images,out_images,test_size=0.20,random_state=42)
    
    in_train = torch.Tensor(in_train) 
    in_valid = torch.Tensor(in_valid)
    out_train = torch.Tensor(out_train) 
    out_valid = torch.Tensor(out_valid)
    
    train_dataset = TensorDataset(in_train,out_train)
    train_dataloader = DataLoader(train_dataset,shuffle=True)
    valid_dataset = TensorDataset(in_valid,out_valid)
    valid_dataloader = DataLoader(valid_dataset,shuffle=True)
    
    return train_dataloader,valid_dataloader
