# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:50:01 2019

@author: Chandrasekar
"""

# Import Libraries
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


training_data = []



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
                    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    training_data.append([np.array(cv2.resize(gray_img, (50,50))), np.eye(2)[gender-1]])


# Shuffle the dataset
np.random.shuffle(training_data)


# Visualize the images and the labels (gender)
# Break the loop when satisified with visualization (Ctrl + c)
for i in range(len(training_data)):
    
    print(training_data[i][1])
    cv2.imshow("frames",training_data[i][0])
    cv2.waitKey(0)

cv2.destroyAllWindows()







# Import the PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a class inherited from the torch nn Module
# We create our conv networks and the FC networks within this class
# When an object of this class is created we essentially create the Conv Network
class Net(nn.Module):
    
    def __init__(self):
        super().__init__() # The constructer for the parent class (nn Module)
        
        # Create the conv layers with the input, output maps and the kernel size as parameters
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,256,3)

        
        # Create a random input x with the dimensions of our image
        # we run this x through our convnets 
        # once we obtain the convoluted output of our x from the final conv layer we can have the dimensions for the First FC layer
        
        # This is nothing but Flattening of the output of final Conv Layer 
        x = torch.rand(50,50).view(-1,1,50,50)
        
        self._to_linear = None

        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))
#        x = F.max_pool2d(F.relu(self.conv5(x)), (2,2))

#        print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

# Define the Forward prop function that sends the input x through all the conv layers and Fc layers and returns the softmax of the final layer
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim = 1)


# Create an object of our class Net
net = Net()


#Importing the torch optim library from which we can use optimzers like Adam and SGD
import torch.optim as optim

# Initialize an optimizer with learning rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)
#Initializing a loss function (Means Squared Loss)
loss_function = nn.MSELoss()

# Splitting the data into train and test sets
x = torch.Tensor([i[0] for i in training_data]).view(-1,1,50,50)
x = x/255.0

y = torch.Tensor([i[1] for i in training_data])

val_split = 0.1

val_size = int(len(x)*val_split)
print(val_size)



train_x = x[:-val_size]

train_y = y[:-val_size]

test_x = x[-val_size:]

test_y = y[-val_size:]


# Initialize the batch size and the number of training epochs
batch_size = 100
epochs = 10

# Run through each epoch
for epoch in range(epochs):
    
    # Running through each batch
    for i in tqdm(range(0, len(train_x), batch_size)):
        
        # Segregating the current training batch
        batch_x = train_x[i:i+batch_size].view(-1,1,50,50)
        batch_y = train_y[i:i+batch_size]

        # We need to zero out the gradients to remove accumulated gradients
        net.zero_grad()
        
        # We pass the input batch as a parameter to the net object of our class
        # This inherently calls our forward prop function
        outputs = net(batch_x)
        
        # Once we have the outputs we can measure the loss
        loss = loss_function(outputs, batch_y)
        
        # We now calculate the gradients
        loss.backward()
        # Our optimizer now uses the loss and gradients to update the weights (one step)
        optimizer.step()
        
        
    print(loss)




# We evaluate the performance of our trained network on the validation set
correct = 0
total = 0

with torch.no_grad():
    
    for i in tqdm(range(len(test_x))):
        real_class = torch.argmax(test_y[i])
        
        net_out = net(test_x[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        
        if predicted_class == real_class :
            correct+=1
        total+=1
        
print("Accuracy : ", round(correct/total,3))


