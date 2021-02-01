
"""

UNET with Spatial transformer - Example training script

"""

from Preprocessing_Functions import *
from Unet_with_spatial_transformer import *
from losses import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import copy



def Validation(valid_loader,model_state,NCC):
    
    device = torch.device("cpu")
    inshape = [112,112]
    model = VxmDense(inshape)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    loss_smooth = MSE().loss
    loss_sim = Grad().loss
    loss_array = []
    corr_array = []
    
    with torch.no_grad():
        
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs,labels)
            L_smooth = loss_smooth(labels,outputs)
            L_sim = loss_sim(labels,outputs)
            L_us = L_sim + weight*L_smooth
       
            NCC_value = NCC(labels,outputs)
       
            loss_array.append(L_us.detach().numpy())
            corr_array.append(NCC_value.detach().numpy())
            
    return np.mean(loss_array),np.mean(corr_array)



# Loading data 

base_pth = "C:/Users/sog19/Desktop/Sheep MR data/bx_sheep3_position3_noHeating/1_Magnitude"
heated_pth = "C:/Users/sog19/Desktop/Sheep MR data/Train Data_sheep"
base_images = Get_DICOM_FILES(base_pth)
heated_images = Get_DICOM_FILES(heated_pth)

print('Image Processing Done')

inshape = [112,112]
device = torch.device("cpu")
net = VxmDense(inshape)
net.to(device)

trainloader,valid_loader = Get_Loader_Valid(heated_images,base_images)

#Training 

NCC = NCC().loss
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_smooth = MSE().loss
loss_sim = Grad().loss
weight = 1
num_epochs = 50

loss_array_train = []
corr_array_train = []
loss_array_valid = []
corr_array_valid = []
best_loss = 100
best_model_state = None

for epoch in range(num_epochs):

    current_loss_array_train = []
    current_corr_array_train = []
    current_loss_array_valid = []
    current_corr_array_valid = []
    
    for i, data in enumerate(trainloader):
        
    
   # get the inputs; data is a list of [inputs, labels]
   
       inputs, labels = data
       inputs, labels = inputs.to(device), labels.to(device)
   # zero the parameter gradients
   
       optimizer.zero_grad()

   # forward + backward + optimize
   
       outputs = net(inputs,labels)
       L_smooth = loss_smooth(labels,outputs)
       L_sim = loss_sim(labels,outputs)
       L_us = L_sim + weight*L_smooth
       
       NCC_value = NCC(labels,outputs)
       
       current_loss_array_train.append(L_us.detach().numpy())
       current_corr_array_train.append(NCC_value.detach().numpy())
       
       current_model_state = copy.deepcopy(net.state_dict())
       
       curr_valid_loss,curr_valid_corr = Validation(valid_loader,current_model_state,NCC)
       
       current_loss_array_valid.append(curr_valid_loss)
       current_corr_array_valid.append(curr_valid_corr)
       
       optimizer.zero_grad()
       L_us.backward()
       optimizer.step()

    loss_array_train.append(np.mean(current_loss_array_train))
    corr_array_train.append(np.mean(current_corr_array_train))
    loss_array_valid.append(np.mean(current_loss_array_valid))
    corr_array_valid.append(np.mean(current_corr_array_valid))
    
    if np.mean(current_loss_array_valid) < best_loss:
        
        best_model_state  = current_model_state
    
    print('TRAIN - Epoch: ', epoch,' loss: ',np.mean(current_loss_array_train), ' Corr: ',
          np.mean(current_corr_array_train))

    print('VALID - Epoch: ', epoch,' loss: ',np.mean(current_loss_array_valid), ' Corr: ',
          np.mean(current_corr_array_valid))
           
np.save('PRELIM_train_loss', np.asarray(loss_array_train))
np.save('PRELIM_train_corr', np.asarray(corr_array_train))
np.save('PRELIM_valid_loss', np.asarray(loss_array_valid))
np.save('PRELIM_valid_corr', np.asarray(corr_array_valid))
torch.save(best_model_state, 'PRELIM_best_model_state')