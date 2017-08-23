#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:39:44 2017

@author: rdelaviz
"""

import os 
os.chdir("/home/rdelaviz/Learning/DL/FastAI/courses/deeplearning1/nbs") 


path = "/home/rdelaviz/Learning/DL/FastAI/courses/deeplearning1/nbs/data/kaggleData/"

from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt


import utils; reload(utils)
from utils import plots

batch_size=32

# Import our class, and instantiate
import vgg16; reload(vgg16)
from vgg16 import Vgg16

vgg = Vgg16(path)

#vgg.model.weights[0]
#vgg.model.load_weights("/home/rdelaviz/.keras/models/vgg16.h5")
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
train_batches = vgg.get_batches(path+'train', batch_size=batch_size)

val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(train_batches)
vgg.model.optimizer.lr = 0.01
vgg.fit(train_batches, val_batches, nb_epoch=10)
#vgg.model.save_weights("RahimModels/Les1Epoch10.h5")
vgg.model.load_weights("RahimModels/Les1Epoch10.h5")

test_batches = vgg.get_batches(path+"test", shuffle=False, batch_size=batch_size, class_mode=None)

preds = vgg.model.predict_generator(test_batches, test_batches.nb_sample)
    
#batches, preds = vgg.test(path+"test", batch_size = batch_size*2)   
import re
imageIds = np.array([ int(re.sub('[tmp\/|\.jpg]','',fName )) for fName in test_batches.filenames])
probIsDog = preds[:,1] # The second column contains the probablity of being a dog.
probIsDog= probIsDog.clip(min=0.0125, max=0.9875)
subm = np.stack([imageIds,probIsDog],axis=1)
import pandas as pd
df = pd.DataFrame(subm)
df.columns=["id","label"]
df = df.sort_values(["id"])
subm2 = np.asarray(df)
submission_file_name="data/kaggleData/epoch10_rah_cat_dog_clip9875.csv"
np.savetxt(submission_file_name, subm2, fmt='%d,%.5f', header='id,label', comments='')

#from PIL import Image 
#Image.open(path+"test/"+test_batches.filenames[100])

clip25 = preds[:,1].clip(min=0.025,max=0.975)
#clip5 = preds[:,1].clip(min=0.05,max=0.95)

#from matplotlib import pyplot as plt
#xx = range(0,len(clip25))
#p = plt.plot(xx,clip5)

#import numpy as np
#import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
#t = np.arange(0, len(clip25), 1)

# red dashes, blue squares and green triangles
#plt.figure(figsize=(13,15))
#plt.plot(clip5,'r', clip25,'g')
#plt.show()
max(clip25)




