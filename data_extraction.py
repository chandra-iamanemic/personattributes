# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:30:28 2019

@author: Chandrasekar
"""

# Importing Libraries
import os
import numpy as np
from shutil import copyfile

path = "C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/Market-1501-v15.09.15/bounding_box_test"
dest = "C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/images_test"
for root,dirs,files in os.walk(path):
    
    for f in files:
        
        #Splitting the file names based on _ and accessing the first element of the split to determine the ID
        ids = f.split('_')
        f_path = os.path.join(path,f)
        dest_path = os.path.join(dest,ids[0])
        
        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)
            
        copyfile(f_path,dest_path + '/' + f)
        
        

path = "C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/Market-1501-v15.09.15/bounding_box_train"
dest = "C:/Users/Chandrasekar/Documents/signatrix test/Market-1501_Attribute-master/images_train"

for root,dirs,files in os.walk(path):
    
    for f in files:

        #Splitting the file names based on _ and accessing the first element of the split to determine the ID        
        ids = f.split('_')
        f_path = os.path.join(path,f)
        dest_path = os.path.join(dest,ids[0])
        
        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)
            
        copyfile(f_path,dest_path + '/' + f)