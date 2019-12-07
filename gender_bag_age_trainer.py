# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:21:46 2019

@author: Chandrasekar
"""

import numpy as np
import os
from scipy.io import loadmat
import cv2
import glob
from tqdm import tqdm

# Load the .mat file containing the attributes for the images
data = loadmat("C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/market_attribute.mat")


   
   
# Path to folder with training images 
train_path = "C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/images_train"

total_data = []



# Run through all the images which are in folders with respect to their IDs and read the images 
# While we read the images from the folders we also compare the IDs of the images to the IDs in the attributes
# The attributes for the images are extracted and the label for the images are formed
# In this case we extract the Gender, Backpack and Age attributes for the images and form a dataset

for i in tqdm(range(len(data['market_attribute'][0][0][1][0][0][0][0]))):
    
#    print(data['market_attribute'][0][0][1][0][0][27][0][i])
    
    person_id = data['market_attribute'][0][0][1][0][0][27][0][i]
    
    gender = data['market_attribute'][0][0][1][0][0][26][0][i]
    
    bag = data['market_attribute'][0][0][1][0][0][1][0][i]

    age = data['market_attribute'][0][0][1][0][0][0][0][i]
    
    gender_one_hot = np.array(np.eye(2)[gender-1]).reshape(1,2)
    bag_one_hot = np.array(np.eye(2)[bag-1]).reshape(1,2)
    age_one_hot = np.array(np.eye(4)[age-1]).reshape(1,4)
    
    gender_one_hot = gender_one_hot.squeeze()
    bag_one_hot = bag_one_hot.squeeze()
    age_one_hot = age_one_hot.squeeze()
    
    np.array([gender_one_hot, bag_one_hot, age_one_hot])
    
    combined_one_hot = [gender_one_hot, bag_one_hot, age_one_hot]
    
    combined_one_hot = np.concatenate( combined_one_hot, axis=0 )

    
    for person in os.listdir(train_path):
        
        if(person==person_id):
            
                os.chdir(os.path.join(train_path,person))
                
                images = [cv2.imread(pic) for pic in glob.glob("*.jpg")]
                
                for image in images:
                    
                    total_data.append([np.array(cv2.resize(image, (224,224))).transpose(2,0,1), combined_one_hot])



# Shuffle the dataset

np.random.shuffle(total_data)

training_data = []
test_data = []


# Split data into train and test sets

for i in range(len(total_data)):
    
    if i%10 == 0:
        test_data.append([total_data[i][0],total_data[i][1]])
    else:
        training_data.append([total_data[i][0],total_data[i][1]])


#for i in range(len(training_data)):
#    
#    print(training_data[i][1])
#    cv2.imshow("frames",training_data[i][0].transpose(1,2,0))
#    cv2.waitKey(0)
#
#cv2.destroyAllWindows()
#

###########################################################




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


train_loader = iter(data_loader)


x,y = next(train_loader)


print(x.shape, y.shape)

print(y)


# Enable CUDA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load a pre trained resnet 34 model 

model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features

# Since we are training for three attributes which I have vectorized as a (1,8) vector 
# The first two elements of this vector are the one hot vector of the gender attribute (binary)
# The second two elements of this vector are the one hot vector of the backpack attribute (binary)
# The last 4 elements of this vector are the one hot vector of the Age attribute (4 classes and hence 4 element one hot vector)

# Hence the final layer must have 8 nodes to train on this output label of 8 elements

model.fc = nn.Linear(num_ftrs, 8)


for name, child in model.named_children():
    print(name)


# We freeze the initial layers to not evaluate its gradients and weights during training
# We unfreeze the final two layers since we want only to train the weights in those layers

for name, child in model.named_children():
    if name in ['layer3', 'layer4']:
        print(name + 'has been unfrozen.')
        for param in child.parameters():
            param.requires_grad = True
    else:
        for param in child.parameters():
            param.requires_grad = False


model = model.to(device)

# We select the MSELoss Function
loss_function = nn.MSELoss()

# We create an optimzer (Stochastic Gradient Descent) which would make the steps in the backprop
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 3 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)



model.train()


epochs = 10

# Run through epochs and perform the forward and backward props for each batch

for epoch in tqdm(range(epochs)):

    
    for inputs, labels in data_loader:
        
        if(labels.shape[0] == 4):
            inputs=inputs.float()
            
            labels.reshape(4,8)
            inputs = inputs.to(device)
            
            labels = labels.float()
            labels = labels.to(device)
                        
            optimizer.zero_grad()
            outputs = model(inputs)
        
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
        else:
            break
        
    print("\n Loss after last epoch is :", loss)


# Save the weights

torch.save(model.state_dict(), "C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/weights/gender_bag_age.pt")


# Evaluate the model on our test set

model.eval()


test_x = torch.Tensor([i[0] for i in test_data]).view(-1,3,224,224)


test_y = torch.Tensor([i[1] for i in test_data])


test_x = test_x.float()
test_y = test_y.float()
test_x = test_x.to(device)
test_y = test_y.to(device)

pred = model(test_x[0:1])

correct_pred_gender = 0
correct_pred_bag = 0
correct_pred_age = 0
total_pred = 0


# Run through the test set and evaluate the accuracy for each of the Attribute (Gender, Bag and Age)

for i in tqdm(range(len(test_x))):
    
    pred = model(test_x[i:i+1])
    actual = test_y[i:i+1].reshape(1,8)
    
    predicted_class_gender = torch.argmax(pred[:,0:2])
    predicted_class_bag = torch.argmax(pred[:,2:4])
    predicted_class_age = torch.argmax(pred[:,4:8])

    actual_gender = torch.argmax(actual[:,0:2])
    actual_bag = torch.argmax(actual[:,2:4])
    actual_age = torch.argmax(actual[:,4:8])

    predicted_class_gender = predicted_class_gender.float()
    predicted_class_bag = predicted_class_bag.float()
    predicted_class_age = predicted_class_age.float()
    
    actual_gender = actual_gender.float()
    actual_bag = actual_bag.float()
    actual_age = actual_age.float()
    
    if predicted_class_gender == actual_gender:
        correct_pred_gender += 1
    if predicted_class_bag == actual_bag:
        correct_pred_bag += 1
    if predicted_class_age == actual_age:
        correct_pred_age += 1
    total_pred +=1
    

print("Accuracy Gender :", round(correct_pred_gender/total_pred,3))
print("Accuracy Bag :", round(correct_pred_bag/total_pred,3))
print("Accuracy Age :", round(correct_pred_age/total_pred,3))


        
    
    
    


