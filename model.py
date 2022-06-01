#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nilearn


# In[51]:


from azureml.core import Environment
from azureml.core import Workspace
curated_env_name = 'AzureML-TensorFlow-2.2-GPU'
ws = Workspace.from_config()
tf_env = Environment.get(workspace=ws, name=curated_env_name)
tf_env = Environment.from_conda_specification(name='tensorflow-2.2-gpu', file_path='AzureML-TensorFlow-2.2-GPU/conda_dependencies.yml')

# Specify a GPU base image
tf_env.docker._enabled = True
tf_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'


# In[31]:


tf_env.save_to_directory(path=curated_env_name)


# In[9]:


pip install nipype


# In[32]:


pip install SimpleITK


# In[33]:


pip install tensorflow==2.2.1


# In[34]:


pip install q keras==2.3.1


# In[35]:


pip install segmentation_models


# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nibabel.testing import data_path
import os
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import segmentation_models as sm
from segmentation_models.metrics import iou_score
focal_loss = sm.losses.cce_dice_loss
from nipype.interfaces.slicer.filtering.n4itkbiasfieldcorrection import N4ITKBiasFieldCorrection
from nipype import Node, Workflow
from nilearn.image import resample_img
from sklearn.model_selection import train_test_split
import random
import SimpleITK as sitk
from nilearn.image import math_img
from nilearn import image as nii
from nilearn import plotting
from nipype.interfaces.ants import N4BiasFieldCorrection


# In[2]:


import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate,Dropout
from tensorflow.keras.layers import Multiply, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[42]:


# import tarfile
# file = tarfile.open('NFBS_Dataset.tar.gz')
# file.extractall()
# file.close()

import zipfile
my_zipfile = zipfile.ZipFile("NFBS_Dataset.zip", mode='r') 
my_zipfile.extractall()
my_zipfile.close()


# In[3]:


print('Each folder contains..')
print(os.listdir('NFBS_Dataset/A00028185'))


# In[4]:


img=nib.load('NFBS_Dataset/A00028185/sub-A00028185_ses-NFB3_T1w.nii.gz')
print('Shape of image=',img.shape)


# In[5]:


import os
brain_mask=[]
brain=[]
raw=[]
for subdir, dirs, files in os.walk('NFBS_Dataset'):
    for file in files:
        #print os.path.join(subdir, file)y
        filepath = subdir + os.sep + file

        if filepath.endswith(".gz"):
          if '_brainmask.' in filepath:
            brain_mask.append(filepath)
          elif '_brain.' in filepath:
            brain.append(filepath)
          else:
            raw.append(filepath)


# In[6]:


data=pd.DataFrame({'brain_mask':brain_mask,'brain':brain,'raw':raw})
data.head()


# In[ ]:


for i in range(5):
  fig,ax=plt.subplots(1,3,figsize=(14,10))
  ax[0].set_title('Raw image')
  img = nib.load(data.raw.iloc[i]).get_data()
  ax[0].imshow(img[img.shape[0]//2])
  ax[1].set_title('Skull strippedimage')
  img = nib.load(data.brain.iloc[i]).get_data()
  ax[1].imshow(img[img.shape[0]//2])
  ax[2].set_title('Brain mask image')
  img = nib.load(data.brain_mask.iloc[i]).get_data()
  ax[2].imshow(img[img.shape[0]//2])


# In[9]:


from N4BiasFieldCorrection.N4BiasFieldCorrection import N4
class preprocessing():
  def __init__(self,df):
    self.data=df
    self.raw_index=[]
    self.mask_index=[]
  def bias_correction(self):
    os.mkdir('bias_correction') 
    # input_image = 
    # output_image= 
    # n4 = N4(input_image, output_image)
    # n4.inputs.dimension = 3
    # n4.inputs.shrink_factor = 3
    # n4.inputs.n_iterations = [20, 10, 10, 5]
    index_corr=[]
    for i in tqdm(range(len(self.data))):
      input_image = self.data.raw.iloc[i]
      output_image ='bias_correction/'+str(i)+'.nii.gz'
      index_corr.append('bias_correction/'+str(i)+'.nii.gz')
      res = N4(input_image,output_image)
    index_corr=['bias_correction/'+str(i)+'.nii.gz' for i in range(11)]
    data['bias_corr']=index_corr
    print('Bias corrected images stored at : bias_correction/')
  def resize_crop(self):
    #Reducing the size of image due to memory constraints
    os.mkdir('resized') 
    target_shape = np.array((96,128,160))                   #reducing size of image from 256*256*192 to 96*128*160
    new_resolution = [2,]*3
    new_affine = np.zeros((4,4))
    new_affine[:3,:3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3,3] = target_shape*new_resolution/2.*-1
    new_affine[3,3] = 1.
    raw_index=[]
    mask_index=[]
    #resizing both image and mask and storing in folder
    for i in range(len(data)):
      downsampled_and_cropped_nii = resample_img(self.data.bias_corr.iloc[i], target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
      downsampled_and_cropped_nii.to_filename('resized/raw'+str(i)+'.nii.gz')
      self.raw_index.append('resized/raw'+str(i)+'.nii.gz')
      downsampled_and_cropped_nii = resample_img(self.data.brain_mask.iloc[i], target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
      downsampled_and_cropped_nii.to_filename('resized/mask'+str(i)+'.nii.gz')
      self.mask_index.append('resized/mask'+str(i)+'.nii.gz')
    return self.raw_index,self.mask_index
  def intensity_normalization(self):
    for i in self.raw_index:
      image = sitk.ReadImage(i)
      resacleFilter = sitk.RescaleIntensityImageFilter()
      resacleFilter.SetOutputMaximum(255)
      resacleFilter.SetOutputMinimum(0)
      image = resacleFilter.Execute(image)
      sitk.WriteImage(image,i)
    print('Normalization done. Images stored at: resized/')


# In[12]:


pre=preprocessing(data)
pre.bias_correction()
r_ind,g_ind=pre.resize_crop()
pre.intensity_normalization()


# In[ ]:


class model():
  def __init__(self,):
    pass
  def split(self,resized_img,resized_mask):
    self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(resized_img,resized_mask,test_size=0.1)
    return self.X_train,self.X_test,self.y_train,self.y_test

  def data_gen(self,img_list, mask_list, batch_size):
 
    c = 0
    n = [i for i in range(len(img_list))]  #List of training images
    random.shuffle(n)
    
    while (True):
      img = np.zeros((batch_size, 96, 128, 160,1)).astype('float')   #adding extra dimensions as conv3d takes file of size 5
      mask = np.zeros((batch_size, 96, 128, 160,1)).astype('float')

      for i in range(c, c+batch_size): 
        train_img = nib.load(img_list[n[i]]).get_data()
        
        train_img=np.expand_dims(train_img,-1)
        train_mask = nib.load(mask_list[n[i]]).get_data()

        train_mask=np.expand_dims(train_mask,-1)

        img[i-c]=train_img
        mask[i-c] = train_mask
      c+=batch_size
      if(c+batch_size>=len(img_list)):
        c=0
        random.shuffle(n)

      yield img,mask

  def convolutional_block(input, filters=3, kernel_size=3, batchnorm = True):
    
    x = Conv3D(filters = filters, kernel_size = (kernel_size, kernel_size,kernel_size),
               kernel_initializer = 'he_normal', padding = 'same')(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters = filters, kernel_size = (kernel_size, kernel_size,kernel_size),
               kernel_initializer = 'he_normal', padding = 'same')(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    return x
  
  #Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
    x = convolutional_block(input, num_filters)
    p = MaxPool3D((2, 2, 2))(x)
    return x, p   

#Decoder block for autoencoder (no skip connections)
def decoder_block(input, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=2, padding="same",activation='relu',kernel_initializer='he_normal')(input)
    x = convlutional_block(x, num_filters)
    return x


#We are getting both conv output and maxpool output for convenience.
def build_encoder(input_image):
    #inputs = Input(input_shape)

    s1, p1 = encoder_block(input_image, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    encoded = convolutional_block(p4, 1024) #Bridge
    
    return encoded

#Decoder for Autoencoder ONLY. 
def build_decoder(encoded):
    d1 = decoder_block(encoded, 512)
    d2 = decoder_block(d1, 256)
    d3 = decoder_block(d2, 128)
    d4 = decoder_block(d3, 64)
    
    decoded = Conv3D(3, (3,3,3),kernel_initializer = 'he_normal', padding="same", activation="sigmoid")(d4)
   
    return decoded

#Use encoder and decoder blocks to build the autoencoder. 
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    autoencoder = Model(input_img, build_decoder(build_encoder(input_img)))
    return(autoencoder)


#Decoder block for unet
def decoder_block_for_unet(input, skip_features, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=2, padding="same",activation='relu',kernel_initializer='he_normal')(input)
    x = Concatenate()([x, skip_features])
    x = convolutional_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = convolutional_block(p4, 1024) #Bridge

    d1 = decoder_block_for_unet(b1, s4, 512)
    d2 = decoder_block_for_unet(d1, s3, 256)
    d3 = decoder_block_for_unet(d2, s2, 128)
    d4 = decoder_block_for_unet(d3, s1, 64)

    outputs = Conv3D(1, (1, 1, 2), activation='sigmoid',padding='same')(d4)  

    model = Model(inputs, outputs, name="U-Net")
    print(model.summary())
    return model
 
  def training(self,epochs):
    im_height=96
    im_width=128
    img_depth=160
    epochs=60
    train_gen = data_gen(self.X_train,self.y_train, batch_size = 4)
    val_gen = data_gen(self.X_test,self.y_test, batch_size = 4)
    channels=1
    input_img = Input((im_height, im_width,img_depth,channels), name='img')
    input_shape = input_img.shape
    
    self.model = build_unet(input_shape)
    self.model.summary()
    self.model.compile(optimizer=Adam(lr=1e-1),loss=focal_loss,metrics=[iou_score,'accuracy'])
    #fitting the model
    callbacks=callbacks = [
        ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True, save_weights_only=False)]
    result=self.model.fit(train_gen,steps_per_epoch=16,epochs=epochs,validation_data=val_gen,validation_steps=16,initial_epoch=0,callbacks=callbacks)
 
  def inference(self,img_path):
    
    #applying bias correction
    output_image= img_path 
    n4 = N4BiasFieldCorrection(img_path,output_image)
    n4.inputs.dimension = 3
    n4.inputs.shrink_factor = 3
    n4.inputs.n_iterations = [20, 10, 10, 5]
    n4.inputs.input_image = img_path
    n4.inputs.output_image =img_path
    res = n4.run()

    #resizing and cropping
    target_shape = np.array((96,128,160))                   #reducing size of image from 256*256*192 to 96*128*80
    new_resolution = [2,]*3
    new_affine = np.zeros((4,4))
    new_affine[:3,:3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3,3] = target_shape*new_resolution/2.*-1
    new_affine[3,3] = 1.
    downsampled_and_cropped_nii = resample_img(img_path, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
    downsampled_and_cropped_nii.to_filename(img_path)
    image = sitk.ReadImage(img_path)

    #intensity normalizing
    rescaleFilter = sitk.RescaleIntensityImageFilter()
    rescaleFilter.SetOutputMaximum(255)
    rescaleFilter.SetOutputMinimum(0)
    image = rescaleFilter.Execute(image)
    sitk.WriteImage(image,img_path)

    #getting predictions 
    orig_img=nib.load(img_path).get_data()
    orig_img=np.expand_dims(orig_img,-1)
    orig_img=np.expand_dims(orig_img,0)
    model=keras.models.load_model('new_best.h5',custom_objects={'categorical_crossentropy_plus_dice_loss':focal_loss,'iou_score':iou_score})
    pred_img=model.predict(orig_img)
    pred_img=np.squeeze(pred_img)
    orig_img=nib.load(img_path).get_data()

    #converting prediction to nifti file
    func = nib.load(img_path)
    ni_img = nib.Nifti1Image(pred_img, func.affine)
    nib.save(ni_img, 'output_T1w_brain_mask.nii.gz')
    pred_img=nib.load('output_T1w_brain_mask.nii.gz')

    #creating binary mask and stripping from raw image
    pred_mask = math_img('img > 0.5', img=pred_img)
    crop=pred_mask.get_data()*orig_img

    #plotting outputs
    pred_img=nib.load('output_T1w_brain_mask.nii.gz').get_data()
    fig,ax=plt.subplots(1,3,figsize=(15,10))
    ax[0].set_title('Original image (cropped)')
    ax[0].imshow(orig_img[orig_img.shape[0]//2])
    ax[1].set_title('Predicted image')
    ax[1].imshow(pred_img[pred_img.shape[0]//2])
    ax[2].set_title('Skull stripped image')
    ax[2].imshow(crop[crop.shape[0]//2])

    #converting skull stripped to nifti file
    ni_img = nib.Nifti1Image(crop, func.affine)
    nib.save(ni_img, 'output_T1w_brain.nii.gz')
    print('Predcited files stores as : output.nii.gz ')

  def plotting(self,filepath):
    
    img=nii.mean_img(filepath)
    plotting.view_img(img,bg_img=img)



