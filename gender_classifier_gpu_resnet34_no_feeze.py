# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:21:46 2019

@author: Chandrasekar
"""
# Import Libraries

import numpy as np
import os
from scipy.io import loadmat
import cv2
import glob
from tqdm import tqdm
import time
import copy

# Load the .mat file containing the attributes for the images

data = loadmat("C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/market_attribute.mat")


   
   
# Path to folder with training images 
train_path = "C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/images_train"

total_data = []


# Run through all the images which are in folders with respect to their IDs and read the images 
# While we read the images from the folders we also compare the IDs of the images to the IDs in the attributes
# The attributes for the images are extracted and the label for the images are formed
# In this case we extract only the Gender attribute for the images and form a dataset


for i in tqdm(range(len(data['market_attribute'][0][0][1][0][0][0][0]))):
    
#    print(data['market_attribute'][0][0][1][0][0][27][0][i])
    
    person_id = data['market_attribute'][0][0][1][0][0][27][0][i]
    
    gender = data['market_attribute'][0][0][1][0][0][26][0][i]
    
    for person in os.listdir(train_path):
        
        if(person==person_id):
            
                os.chdir(os.path.join(train_path,person))
                
                images = [cv2.imread(pic) for pic in glob.glob("*.jpg")]
                
                for image in images:
                    
                    total_data.append([np.array(cv2.resize(image, (224,224))).transpose(2,0,1), np.eye(2)[gender-1]])

# Shuffle the dataset

np.random.shuffle(total_data)

training_data = []
test_data = []

# Split the dataset into train and test sets
for i in range(len(total_data)):
    
    if i%10 == 0:
        test_data.append([total_data[i][0],total_data[i][1]])
    else:
        training_data.append([total_data[i][0],total_data[i][1]])




# Import PyTorch Libraries required by us
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


# Create a dataloader with a specified batch size from the training data
data_loader = DataLoader(training_data, batch_size = 4,
                         shuffle = True, num_workers = 0)

# Run through one iteration of the data loader (one batch) and see the batch images/labels and its sizes and shape
test_data_loader = DataLoader(test_data)

train_loader = iter(data_loader)


x,y = next(train_loader)


print(x.shape, y.shape)

print(y)



# Enable CUDA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Load a pre trained resnet 34 model 

model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features

# Since we are classifying gender (0,1) which is binary we change the final FC layer output to 2 nodes
model.fc = nn.Linear(num_ftrs, 2)


# We can see the layers of the network
for name, child in model.named_children():
    print(name)


# We move the model to CUDA 
model = model.to(device)

# We select the MSELoss Function
loss_function = nn.MSELoss()

# We create an optimzer (Stochastic Gradient Descent) which would make the steps in the backprop
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 3 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# Enter the train mode
model.train()


# Initialize the number of epochs
epochs = 6


# Run through the epochs
for epoch in tqdm(range(epochs)):

    
    # Run through a single batch
    for inputs, labels in data_loader:
        inputs=inputs.float() 
        labels = labels.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero out the grads
        optimizer.zero_grad()
        
        # Forward Prop
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
    
        # Calculate loss
        loss = loss_function(outputs, labels)
        
        # Backward Prop
        loss.backward()
        optimizer.step()
        
    print(loss)


# Save the weights    
torch.save(model.state_dict(), "C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/weights/gender_classifier.pt")


# Enter Evaluation mode
model.eval()

# extract the data in test set as Tensors and reshape to the right size
test_x = torch.Tensor([i[0] for i in test_data]).view(-1,3,224,224)


test_y = torch.Tensor([i[1] for i in test_data])


test_x = test_x.float()
test_y = test_y.float()
test_x = test_x.to(device)
test_y = test_y.to(device)

pred = model(test_x[0:1])

correct_pred = 0
total_pred = 0

# Run predictions and calculate the accuracy

for i in tqdm(range(len(test_x))):
    
    pred = model(test_x[i:i+1])
    predicted_class = torch.argmax(pred)
    predicted_class = predicted_class.float()
    
    if predicted_class == test_y[i]:
        correct_pred += 1
    total_pred +=1
    

print("Accuracy :", round(correct_pred/total_pred,3))
        
    
    
    


