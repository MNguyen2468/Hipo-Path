#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
import builtins
import os
# sys.stdout = open("job.txt", "w", buffering=1)
# def print1(text):
#     builtins.print(text)
#     os.fsync(sys.stdout)

print("This is immediately written to stdout.txt")

print("########################started#################################")
import cv2 
import pandas as pd
import numpy as np
import os
import random
from os import walk
from PIL import ImageFile
import seaborn as sns
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
#random.seed(42)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from DataGenetors import ImgDataParameters,DataGenerator
print("########################importedstarted#################################")
# from CAT_Net2 import CATNet2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Attention
from tensorflow.keras import optimizers
# from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("########################tensorflowdone#################################")
print("import done")
# import keras.backend as K
# from keras.callbacks import EarlyStopping, ModelCheckpoint
#The GPU id to use, usually either "0" or "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**96)
# import sys
sys.path.append("..")
from WSI_Preprocessing.Preprocessing.WSI_Scanning import readWSI
from WSI_Preprocessing.Preprocessing.Denoising import denoising
from WSI_Preprocessing.Preprocessing.Patch_extraction_creatia import patch_extraction_random, all_patches_extarction
from WSI_Preprocessing.Preprocessing.Utilities import stainremover_small_patch_remover1
import openslide
from openslide import (OpenSlide, OpenSlideError,OpenSlideUnsupportedFormatError)
# from keract import get_activations
# from keract import display_activations

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
def mean_act(y_true, y_pred):
    return K.mean(y_true)


# In[2]:
print("########################modelstarted#################################")

model = load_model(r'./modelfinal_train_14.h5')
layer_name = 'mixed10'  #I am changing this from conv2d_188 to mixed10 - CG 03142025
intermediate_layer_model = Model(inputs=model.input,
outputs=[model.get_layer(layer_name).output,model.output])
print("########################model1loaded#################################")
#model1 = load_model(r'/home/kosaraju/ALK/ALK_src/newmodel13new.h5',custom_objects={'Attention': Attention})
#layer_name1 = 'add'
#intermediate_layer_model1 = Model(inputs=model1.input,
#outputs=[model1.get_layer(layer_name1).output,model1.output])
print("########################model1loaded#################################")
print("model load done")

# In[3]:


def plot(y_pred):
    plt = sns.heatmap(np.array([[y_pred]]), yticklabels = False,
                      xticklabels = False, cmap='coolwarm', 
                      vmin = 0, vmax = 1, cbar = False).get_figure()
    plt.savefig("example.png")
    img = cv2.imread("example.png")
    img  = np.where(img != [255,255,255],img , img[144,144])
    img = cv2.resize(img, (256,256))
    os.remove("example.png")
    return img

def HipoScore(intermediate_outputH,y_pred):
    print(f"intermediate shape is {intermediate_outputH.shape}")
    al_20 = np.zeros((1,6,6))
    al_201 = np.zeros((1,6,6))
    for t in range(len(intermediate_outputH[0][0][0])):
        grad = np.gradient(intermediate_outputH[:,:,:,t].flatten(),abs(1-y_pred+0.000008))
        al = ReLU(sum(grad))
        al_20 += al*intermediate_outputH[:,:,:,t]
        al_201 += intermediate_outputH[:,:,:,t]
#         print(al_20.shape)
#     print(al_20.shape)
    al_20 = ReLU(al_20)
    return al_20
def HipoScore1(intermediate_outputH,y_pred):
    print(f"intermediate shape is {intermediate_outputH.shape}")
    al_202 = []
    for t in range(len(intermediate_outputH[0][0][0])):
        grad = np.gradient(intermediate_outputH[:,:,:,t].flatten(),abs(1-y_pred+0.000008))
        al = ReLU(sum(grad))
        al_202.append(al*np.sum(intermediate_outputH[:,:,:,t]))
        
#         print(al_20.shape)
#     print(al_20.shape)
    al_202 = al_202
    return np.array(al_202)
def ReLU(x):
    return x * (x > 0)

slide_number = sys.argv[1] if len(sys.argv) >= 2 else ''
def extractingPatches(inputsvs, outputpath, magnification, patch_extraction_creatia = None,num_of_patches = 2000, 
                      filtering = "GaussianBlur", patch_size = (256,256), upperlimit = 900, 
                      lowerlimit = 300, red_value = (80,220), green_value = (80,200), blue_value = (80, 170),  
                      reconstructedimagepath = None, Annotation = None, Annotatedlevel = 0, Requiredlevel = 0, 
                      Requiredlevel1 = 0,model=None):
    
    Y= []
    slide1,slidedim = readWSI(inputsvs, "20x", Annotation, Annotatedlevel, Requiredlevel)
    slide2,slidedim  = readWSI(inputsvs, "5x", Annotation, Annotatedlevel, Requiredlevel1)
    Y_ = []
    Y_s = []
    k = 0 
    k1 = 0 
    ALL_P = []
    #ALL_P = np.asanyarray(ALL_P)
    ALL_P1 = []
    #ALL_P1 = np.asanyarray(ALL_P1)
    ALL_P3 = []
    #ALL_P3 = np.asanyarray(ALL_P3)
    ALL_P2 = []
    all_patchs1 = []
    print("########################loadedmodels#################################")

    for i in range(int(len(slide1[0])/128)):
        for j in range(int(len(slide1)/128)):
            centerpoint20x = ((j*256+(patch_size[0]/2)),(i*256+(patch_size[1]/2)))
            centerpoint5x = (centerpoint20x[0] * (len(slide2) / len(slide1)), centerpoint20x[1] * (len(slide2[0]) / len(slide1[0])))
            sample_img = slide1[int(centerpoint20x[0]-patch_size[0]/2):int(centerpoint20x[0]+patch_size[0]/2),
                         int(centerpoint20x[1] - patch_size[1]/2): int(centerpoint20x[1] + patch_size[1]/2)]
            # sample_img1 = slide2[int(centerpoint5x[0]-patch_size[0]/2):int(centerpoint5x[0]+patch_size[0]/2),
                        #  int(centerpoint5x[1] - patch_size[1]/2): int(centerpoint5x[1] + patch_size[1]/2)]
            
            # centerpoint20x = ((j*299+(patch_size[0]/2)),(i*299+(patch_size[1]/2)))
            # sample_img = slide1[int(centerpoint20x[0]-patch_size[0]/2):int(centerpoint20x[0]+patch_size[0]/2),
            #              int(centerpoint20x[1] - patch_size[1]/2): int(centerpoint20x[1] + patch_size[1]/2)]
            patchs = stainremover_small_patch_remover1(sample_img, patch_size, slide_number)
            
            if patchs is None:
                None
            else:
                # print(i)
                patchs1 = patchs/255

#                 y_pred = model.predict(patchs1)
                al_p = []
                al_p1 = []

                
#                  np.insert(np.array(intermediate_output),0,1,axis =1)
                # patchs1 = patchs/255
                # sample_img1 = sample_img1/255
#                         patchs = np.dot(patchs[:,:,:3], [0.299, 0.587, 0.114])
#                         sample_img1 = np.dot(sample_img1[:,:,:3], [0.299, 0.587, 0.114])
                # sample_img1 = np.expand_dims(sample_img1, axis = 0)
                patchs1 = np.expand_dims(patchs1, axis=0)
                try:
#                   
                    # print(patchs1.shape, sample_img1.shape)
                    #y_pred = model.predict(patchs1)
                    #print(y_pred)
                    # print()
                    y_pred = model.predict(patchs1)
                    # print(y_pred1)S
                    #y_pred1[0][0] = max(y_pred1[0][0] -0.18,0)
                    #print(y_pred1)
                    ALL_P1.append((y_pred[0][0],j,i)) #remove pred1
                    intermediate_output,y_pred2 = intermediate_layer_model.predict(patchs1)
#                     y_pred2[0][0] = max(y_pred2[0][0] - 0.07,0)
                    al_h =  HipoScore(intermediate_output,y_pred[0][0])
                    #al_h1 = HipoScore1(intermediate_output,y_pred[0][0])
                    ALL_P.append(al_h)
                    #ALL_P2.append(al_h1)
#                     /y_pred = y_pred
                    ALL_P3.append((y_pred[0][0],j,i))
                    print(y_pred[0][0])
                    print(y_pred2[0][0])
                    print(np.mean(al_h))
                    #print("cancer:%s,int:%s,HipoScore:%s"(y_pred[0][0],y_pred2[0][0],np.mean(al_h)))
                except Exception as e:
                    #print(sample_img1.shape)
                    print(patchs.shape)
                    print(e)
#                     print(patchs.shape)
                    # print(sample_img1.shape)
#                     print(a)
    print("finished slide: Saving now...")
    return ALL_P1,ALL_P,ALL_P3#,ALL_P2


# In[ ]:


from PIL import Image as im
import csv
path_mod = f'./Slides{slide_number}/'
os.makedirs(path_mod, exist_ok=True)
list_mod = []
for root, dirs, files in os.walk(path_mod):
    for file in files:
        if file.endswith('svs'):
            list_mod.append(os.path.join(root, file))

#os.listdir(path_mod)



# Assuming list_mod contains filenames (strings)
# And you have another directory to compare against:
other_dir = "./HipoScores_t"
os.makedirs(other_dir, exist_ok=True)
other_filenames = os.listdir(other_dir)

# Get set of first 23-character prefixes in other directory
other_prefixes = {fname[:23] for fname in other_filenames}

# Now filter your list_mod
list_slide = []
for file in list_mod:
    prefix = os.path.basename(file)[:23]  
    if prefix not in other_prefixes:
        list_slide.append(file)
    #if i.endswith('svs') and i[:23] not in other_prefixes:
        
        #list_slide.append(i)
# for i in list_mod:
#     if i[-3:] == 'svs':
#         list_slide.append(i)
print(len(list_slide))

#slide_name = 'TCGA-O1-A52J-01Z-00-DX1.26F6ECCA-D614-4950-98E6-4D76E82F71B4.svs'
dict_y_pred = []
dict_y_count = []
for i in list_slide[:]:
    print(i)
    tmp_pred = {}
    tmp_count = {}
    tmp_heatmap = {}
    print(i)
    data = i #path_mod + i
    print(data)
#     model = load_model('newmodel13new.h5',custom_objects={'Attention': Attention})
    slide = OpenSlide(data)
    slide_dimensions = slide.level_dimensions
    if len(slide_dimensions) == 3:
        ALL_P1,ALL_P,ALL_P3 = extractingPatches(data,"temp",magnification = "20x",patch_size= (256,256),Annotation = None,Annotatedlevel = 0, Requiredlevel = 0,Requiredlevel1 = 1,model=model)
    else:
        ALL_P1,ALL_P,ALL_P3 = extractingPatches(data,"temp",magnification = "20x",patch_size= (256,256),Annotation = None,  Annotatedlevel = 0, Requiredlevel = 1,Requiredlevel1 = 2,model=model)

    scores_path = './scores_t'
    os.makedirs(scores_path, exist_ok=True)
    gradcam_20_path1 = os.path.join(scores_path, os.path.basename(i)[:-4] + "new")
    np.save(gradcam_20_path1, ALL_P1)

    hipo_scores_path = './HipoScores_t'
    os.makedirs(hipo_scores_path, exist_ok=True)
    gradcam_20_path2 = os.path.join(hipo_scores_path,  os.path.basename(i)[:-4] + "new")
    np.save(gradcam_20_path2, ALL_P)

    grad_prob_path = './Grad_prob_t'
    os.makedirs(grad_prob_path, exist_ok=True)
    gradcam_20_path3 = os.path.join(grad_prob_path,  os.path.basename(i)[:-4] + "new")
    np.save(gradcam_20_path3, ALL_P3)

    #hipo_scores_t_update_path = './HipoScores_t_update'
    #os.makedirs(hipo_scores_t_update_path, exist_ok=True)
    #gradcam_20_path4 = os.path.join(hipo_scores_t_update_path,  i[:-4] + "new")
    #np.save(gradcam_20_path4, ALL_P2)

    # gradcam_20_path1 = '/home/kosaraju//ALK/New_data_results/Additional_data///////scores/' + '1028201' + "new"
    # gradcam_20_path2 = '/home/kosaraju//ALK/New_data_results/Additional_data//////HipoScores/' + '1028201' + "new"
    # gradcam_20_path3 = '/home/kosaraju//ALK/New_data_results/Additional_data/////Grad_prob/' + '1028201' + "new"
#     gradcam_20_path = '/home/wsai/Newinter/H&E/Mod-diff/' + i[-11:-4]
#     gradcam_20_path = '/home/wsai/Newinter/H&E/Mod-diff/' + i[-11:-4]
#     all_patchs1 = np.sort(np.array(ALL_P), axis=0)[::-1]




#     np.save(gradcam_20_path1, ALL_P1
