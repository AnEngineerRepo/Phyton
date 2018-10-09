# -*- coding: utf-8 -*-
from __future__ import division
import SimpleITK as sitk
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot  import show, plot
plt.switch_backend('Agg')
#%matplotlib inline
import os
import cPickle
import time
import random
from skimage import transform
from random import randint
from skimage.transform import downscale_local_mean
from sklearn import metrics
from sklearn.metrics import r2_score

np.random.seed(0)

def getLargestDims(imagepaths):     
    largestdims = np.zeros(3,dtype=np.int16)
    for i in xrange(len(imagepaths)):
        imtemp = sitk.ReadImage(imagepaths[i]) 
        dims = imtemp.GetSize()
        vs = imtemp.GetSpacing() 
        for j in xrange(len(largestdims)):
            if (dims[j]>largestdims[j]):
                largestdims[j]=dims[j]   
   
    print 'Will use these dimensions: {}'.format(largestdims)    
    return largestdims
    
def loadImages(imagepaths,segmentationfolder,mask_folder,largestdims,nroflabels):
    images = np.array([],dtype=np.float32)
    segs = np.array([],dtype=np.int16)
    brain_masks = np.array([],dtype=np.int16)
    count = 0
    patients = []
    
    print 'Loading the images...'
    for i in xrange(len(imagepaths)):
        filename = os.path.splitext(os.path.basename(imagepaths[i]))[0]
        print filename
        segmentation = filename + 'multiseg'
        mask_name = filename + '_brain_mask'
        print segmentation
        segmentationpath = os.path.join(segmentationfolder,segmentation+'.nii')     
        mask_path = os.path.join(mask_folder,mask_name+'.nii')     
        print segmentationpath
        patients.append(filename)
    
        if os.path.exists(segmentationpath): 
            count += 1
            sitkimage = sitk.ReadImage(imagepaths[i])
            voxelsizes = sitkimage.GetSpacing()
            imagedims = sitkimage.GetSize()                                             
            difference = largestdims - imagedims
            padding = np.swapaxes(np.vstack((np.zeros(3),difference)),0,1).astype(np.int)        
            oimage = np.swapaxes(sitk.GetArrayFromImage(sitkimage),0,2).astype(np.int16)
            image = np.pad(oimage, tuple(map(tuple,padding)), 'constant', constant_values=0)
            sitkseg = sitk.ReadImage(segmentationpath)
            oseg = np.swapaxes(sitk.GetArrayFromImage(sitkseg),0,2).astype(np.int16)
            seg = np.pad(oseg, tuple(map(tuple,padding)), 'constant', constant_values=0)
            sitk_mask = sitk.ReadImage(mask_path)
            omask = np.swapaxes(sitk.GetArrayFromImage(sitk_mask),0,2).astype(np.int16)
            mask = np.pad(omask, tuple(map(tuple,padding)), 'constant', constant_values=0)
                
            if (len(images)>0):            
                images = np.concatenate((images, np.swapaxes(image,0,2))) 
                segs = np.concatenate((segs, np.swapaxes(seg,0,2)))
                brain_masks = np.concatenate((brain_masks, np.swapaxes(mask,0,2)))             
            else:
                images = np.swapaxes(image,0,2)
                segs = np.swapaxes(seg,0,2)
                brain_masks=np.swapaxes(mask,0,2)
    
    print 'Dimensions: {}'.format(images.shape) 
    print 'Dimensions: {}'.format(segs.shape)  
    
    volumeperslice = np.zeros((len(images),nroflabels))
    for n in xrange(nroflabels):
        volumeperslice[:,n] = np.sum(np.sum((segs==n+1).astype(np.int16),axis=2),axis=1)*voxelsizes[0]*voxelsizes[1]*voxelsizes[2]

    print 'Dim: {}'.format(volumeperslice.shape)
    for n in xrange(nroflabels):
        print 'Max for label {}: {}'.format(n+1,np.max(volumeperslice[:,n]))

    images = images.reshape((count*image.shape[2],1,image.shape[0],image.shape[1])) 
    print 'Dimensions: {}'.format(images.shape)
    segs = segs.reshape((count*seg.shape[2],1,seg.shape[0],seg.shape[1])) 
    print 'Dimensions: {}'.format(segs.shape)
    
    return images, volumeperslice, brain_masks, patients, segs

imagepaths = np.array(sorted(glob.glob(r'/home/joao/Code/30wks/Images/*.nii')))
segmentationfolder = r'/home/joao/Code/30wks/Segmentations'
mask_folder = r'/home/joao/Code/30wks/BrainMasks30wks'   

nroflabels = 8
trainnr = 60
testnr = 13
valnr = 13

print '{} images'.format(len(imagepaths)) 

imageindices = np.arange(len(imagepaths))
np.random.shuffle(imageindices)
trainimagepaths = imagepaths[imageindices[:trainnr]]
testimagepaths = imagepaths[imageindices[trainnr:trainnr+testnr]]
valimagepaths = imagepaths[imageindices[trainnr+testnr:trainnr+testnr+valnr]]    

print '{} training images'.format(len(trainimagepaths))
print '{} test images'.format(len(testimagepaths))
print '{} validation images'.format(len(valimagepaths))
print '{} images not used'.format(len(imagepaths)-trainnr-testnr-valnr)
    
largestdims = getLargestDims(imagepaths)             
    
trainimages, trainvolumeperslice,trainmasks,trainpatients, trainsegs = loadImages(trainimagepaths,segmentationfolder,mask_folder,largestdims, nroflabels)
valimages, valvolumeperslice,valmasks,valpatients, valsegs = loadImages(valimagepaths, segmentationfolder,mask_folder,largestdims, nroflabels)
#testimages, testvolumeperslice, testpatients = loadImages(testimagepaths, segmentationfolder, largestdims, nroflabels)

def optimized_per_pic(slice_pic,masks):
        array_h,array_w=[],[]
        min_w,max_w,min_h,max_h=(10000,0,10000,0)
        for h in xrange(masks.shape[1]):
            for w in xrange(masks.shape[2]):
                if masks[slice_pic,h,w]==1:
                    array_h.append(h)
                    array_w.append(w)
        if (array_h!=[]):
            min_h=min(array_h)
            max_h=max(array_h)
            min_w=min(array_w)
            max_w=max(array_w)
        
        return  min_w,max_w,min_h,max_h
    
def optimized_values(images,masks):
    values=[]
    w_real_min,w_real_max,h_real_min,h_real_max=(images.shape[3],0,images.shape[2],0)
    for x in xrange(images.shape[0]):
        values.append(optimized_per_pic(x,masks))
        if values[x][0]<w_real_min:
            w_real_min=values[x][0]
        if values[x][1]>w_real_max:
            w_real_max=values[x][1]
        if values[x][2]<h_real_min:
            h_real_min=values[x][2]
        if values[x][3]>h_real_max:
            h_real_max=values[x][3]
    return w_real_min,w_real_max,h_real_min,h_real_max   

train_values=optimized_values(trainimages,trainmasks)
val_values=optimized_values(valimages,valmasks)

min_width=min(train_values[0],val_values[0])
max_width=max(train_values[1],val_values[1])
min_height=min(train_values[2],val_values[2])
max_height=max(train_values[3],val_values[3])
print(min_width,max_width,min_height,max_height)

def reduced_pic(images,values):
    reduced_pic=[]
    for z in xrange(images.shape[0]):
        reduced_pic.append(images[z,0,values[2]:values[3],values[0]:values[1]])
    np_reduced_pic=np.array(reduced_pic)
    print(np_reduced_pic.shape)
    return np_reduced_pic
 
width=max_width-min_width
height=max_height-min_height
print(width, height)
 
reduced_trainimages=reduced_pic(trainimages,train_values)
reduced_valimages=reduced_pic(valimages, train_values)

print(reduced_trainimages.shape)
def downscale (images,factor):
    subsample=[]
    for n in xrange(images.shape[0]):
        downscaled=downscale_local_mean(images[n], (factor, factor))
        subsample.append(downscaled)
    np_subsample=np.asarray(subsample)
    reshaped_np_subsample=np.reshape(np_subsample,(np_subsample.shape[0],1,np_subsample.shape[1],np_subsample.shape[2]))
    return reshaped_np_subsample

reshaped_np_subsample_train=downscale(reduced_trainimages,2)
reshaped_np_subsample_val=downscale(reduced_valimages,2)
print(reshaped_np_subsample_train.shape)
print(reshaped_np_subsample_val.shape)
height=min(reshaped_np_subsample_train.shape[2],reshaped_np_subsample_val.shape[2])
width=min(reshaped_np_subsample_train.shape[3],reshaped_np_subsample_val.shape[3])
print(height,width)

print(reshaped_np_subsample_train.shape)
def turn3D(downscaled_images, depth):
    array=[]
    for x in xrange(int(downscaled_images.shape[0]/depth)):
        array.append(downscaled_images[depth*x:depth*x+depth])
    return array

deepness=9
array_train=turn3D(reshaped_np_subsample_train, deepness)
print(array_train[0].shape)
print(len(array_train))
array_val=turn3D(reshaped_np_subsample_val, deepness)

print("------------------------------CNN code-------------------------------")

import lasagne
import theano
from theano import tensor as T
#from lasagne.layers import dnn

def residual_block(layer, num_filters, filter_size=3, stride=1, num_layers=2):
    if (num_filters != layer.output_shape[1]) or (stride != 1):
        layer = lasagne.layers.Conv3DLayer(layer, num_filters, filter_size=1, stride=stride, pad=0, nonlinearity=lasagne.nonlinearities.rectify, b=None)
	#layer=lasagne.layers.batch_norm(layer)
    conv = layer
    for _ in range(num_layers):
        conv = lasagne.layers.Conv3DLayer(conv, num_filters, filter_size,pad='same', nonlinearity=lasagne.nonlinearities.rectify)
	#conv=lasagne.layers.batch_norm(conv)
    return lasagne.layers.batch_norm(lasagne.layers.ElemwiseSumLayer([conv, layer]))

def getmodel(X1,p_drop_hidden, height, width, deepness): 
             
    layer = lasagne.layers.InputLayer(shape=(None, 1,deepness, height, width),input_var=X1)
    print(layer.output_shape)
    layer = lasagne.layers.Conv3DLayer(layer, 16, 7, stride=2, pad='same', nonlinearity=lasagne.nonlinearities.rectify)
    layer=lasagne.layers.batch_norm(layer)
    print(layer.output_shape)
    layer = lasagne.layers.Pool3DLayer(layer, 2)
    print(layer.output_shape)
    for _ in range(3):
        layer = residual_block(layer, 16)
    print(layer.output_shape)
    layer = residual_block(layer, 32, stride=2)
    print(layer.output_shape)
    for _ in range(3):
        layer = residual_block(layer, 32)
    print(layer.output_shape)
    layer = residual_block(layer, 64, stride=2)
    print(layer.output_shape)
    for _ in range(5):
        layer = residual_block(layer, 64)
    print(layer.output_shape)
    layer = residual_block(layer, 128, stride=2)
    print(layer.output_shape)
    for _ in range(2):
        layer = residual_block(layer, 128)
    print(layer.output_shape)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(layer, p=p_drop_hidden),num_units=256,nonlinearity=lasagne.nonlinearities.rectify)
    print(network.output_shape)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=p_drop_hidden),num_units=8,nonlinearity=lasagne.nonlinearities.linear)
    print(network.output_shape)
    return network   
       
X1 = T.tensor5()  
Y = T.matrix()   

model= getmodel(X1, 0.5, height, width, deepness)
py_x_noise = lasagne.layers.get_output(model) 
cost = T.mean(lasagne.objectives.squared_error(py_x_noise, Y))
params = lasagne.layers.get_all_params(model, trainable=True)
updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.001)

train = theano.function(inputs=[X1, Y], outputs=cost, updates=updates, allow_input_downcast=True)

# Make classification functions
outputprediction = lasagne.layers.get_output(model, deterministic=True)
predict = theano.function(inputs=[X1], outputs=outputprediction, allow_input_downcast=True)

#convoutput = lasagne.layers.get_output(convmodel, deterministic=True)
#get_convoutput = theano.function(inputs=[X1],outputs=convoutput, allow_input_downcast=True)

costvalidation = T.mean(lasagne.objectives.squared_error(outputprediction, Y))
validation = theano.function(inputs=[X1, Y], outputs=costvalidation, allow_input_downcast=True)

def sum_labels(volumes, depth):
    sum_volumes=[]
    for i in xrange(8):
        k=0
        sum_tissue=[]
        while k<volumes.shape[0]/depth:
            sum_tissue.append((np.sum(volumes[depth*k:depth*k+depth,i])))
            k=k+1
        sum_volumes.append(sum_tissue)
    np_sum_volumes=np.asarray(sum_volumes)
    np_sum_volumes_order=np.swapaxes(np_sum_volumes,0,1)
    return(np_sum_volumes_order)

train_sum_labels=sum_labels(trainvolumeperslice, deepness)
val_sum_labels=sum_labels(valvolumeperslice, deepness)

print(train_sum_labels.shape)
print(val_sum_labels.shape)

def normalization(labels,trainlabels):
    array_tutti, train_mins, train_maxs=[],[],[]
    for h in xrange(8):
        minimo=min(trainlabels[:,h])
        maximo=max(trainlabels[:,h])
        train_mins.append(minimo)
        train_maxs.append(maximo)
        array_tutti.append((labels[:,h:h+1]-minimo)/(maximo-minimo))
    np_normalized=np.asarray(array_tutti)
    reshaped_np_normalized=np.reshape(np_normalized,(8,labels.shape[0]))
    ordered_np_normalized=np.swapaxes(reshaped_np_normalized,0,1)
    return(ordered_np_normalized)
ttt_volumes=normalization(train_sum_labels,train_sum_labels)
vvv_volumes=normalization(val_sum_labels,train_sum_labels)
print(ttt_volumes.shape)
print(vvv_volumes.shape)

def convert_to_np(big_array, imagesample):
    t=[]
    for e in xrange(len(imagesample)):
        t.append(big_array[imagesample[e]])
    np_array=np.array(t)
    np_new=np_array.reshape((np_array.shape[0],np_array.shape[2],np_array.shape[1],np_array.shape[3],np_array.shape[4]))
    return(np_new)

def transform2Dimage(image2D,angle,scale,tx,ty): #angle: degrees, scale: factor, tx ty: voxels
    halfsize = image2D.shape[0]/2
    T = transform.AffineTransform(translation=(halfsize,halfsize))
    RS = transform.AffineTransform(rotation=angle/360*2*np.pi,scale=(1/scale,1/scale))
    Tinv = transform.AffineTransform(translation=(-halfsize+tx,-halfsize+ty))
    transfmatrix = np.dot(np.dot(T.params,RS.params),Tinv.params)
    rotatedimage = transform.warp(image2D,transfmatrix)
    return rotatedimage

def ChangeIntensity(np_image,number,f_number):
    np_new_img=np_image**number
    if f_number==1:
        np_new_img=np.flip(np_new_img,1)
    return np_new_img

def WholeTransformation(batch_images, batch_volumes): 
    transform_matrix=[]
    all_volumes=[]
    for k in (xrange(batch_images.shape[0])):
        for j in (xrange(20)):
            transf_chunk=[]
            scale_factor=random.uniform(0.8, 1)
            angle=randint(-15,15)
            tx=randint(0,0)
            ty=randint(0,0)
            intensity_number=np.random.uniform(0.95, 1.05)
            flip_number=randint(0,1)
            for kk in xrange(batch_images.shape[2]):
                change = transform2Dimage(batch_images[k,0,kk],angle,scale_factor,tx,ty)
                i_change=ChangeIntensity(change,intensity_number,flip_number)
                transf_chunk.append(i_change)
            np_transf_chunk=np.asarray(transf_chunk)     
            transform_matrix.append(np_transf_chunk)
            new_volume=batch_volumes[k]*(scale_factor*scale_factor)
            all_volumes.append(new_volume)
    np_transform_matrix = np.array(transform_matrix)
    #np_all_volumes = np.array(all_volumes)
    reshaped_np_transform_matrix=np.reshape(np_transform_matrix,(np_transform_matrix.shape[0],1,np_transform_matrix.shape[1],np_transform_matrix.shape[2],np_transform_matrix.shape[3]))
    full_batch_images=np.concatenate((batch_images,reshaped_np_transform_matrix),axis=0)
    full_batch_volumes=np.concatenate((batch_volumes,all_volumes),axis=0)
    
    return full_batch_images, full_batch_volumes
	
minibatches = 20000
minibatchsize = 2

losslist = []
validlosslist = []
    
trainingsamples = np.arange(len(array_train)) #numbers from 0 until the number of samples
validsamples = np.arange(len(array_val))
print("I am training....")
t0 = time.time()
for i in xrange(minibatches):
    imagesample=random.sample(trainingsamples,minibatchsize)
    t_images=convert_to_np(array_train, imagesample)
    t_volumes=ttt_volumes[imagesample]
    full_t_images,full_t_volumes=WholeTransformation(t_images,t_volumes)
    x=train(full_t_images,full_t_volumes)
    losslist.append(x)
    validimagesample=random.sample(validsamples,minibatchsize)
    v_images=convert_to_np(array_val, validimagesample)
    v_volumes=vvv_volumes[validimagesample]
    full_v_images,full_v_volumes=WholeTransformation(v_images,v_volumes)
    y=validation(full_v_images,full_v_volumes)
    validlosslist.append(y)
    if  i==1000 or i==2000 or i==3000 or i==4000 or i==5000 or i==6000 or i==7000 or i==8000 or i==9000 or i==10000 or i==11000 or i==12000:
        print("Mais 1000")

t1 = time.time()
print 'Training time: {} seconds'.format(t1-t0)

plt.figure()
plt.plot(losslist)
plt.plot(validlosslist)
plt.yscale("log")
plt.savefig('3D_54_validlosslist')
plt.figure()
plt.plot(losslist)
plt.yscale("log")
plt.savefig('3D_54_trainloss')

validsamples = np.arange(len(array_val))
validation_sample=random.sample(validsamples,13)
validation_images=convert_to_np(array_val, validation_sample)
validation_volumes=vvv_volumes[validation_sample]
print(validation_volumes.shape)
print(validation_images.shape)
augmented_val_images,augmented_val_volumes=WholeTransformation(validation_images, validation_volumes)

z1=predict(augmented_val_images[:35])
z2=predict(augmented_val_images[35:70])
z3=predict(augmented_val_images[70:105])
z4=predict(augmented_val_images[105:140])
z5=predict(augmented_val_images[140:175])
z6=predict(augmented_val_images[175:210])
z7=predict(augmented_val_images[210:245])
z8=predict(augmented_val_images[245:273])

predicted_results=np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8),axis=0)
print(predicted_results.shape)

def save_weights(filename,network):
    with open(filename, 'wb') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(network), f)
def save_volumes(filename, volumes):
    with open(filename, 'wb') as f:
        cPickle.dump(volumes, f)        

save_weights('/home/joao/Code/ModelsVolumes/[TRACKING+3D+AUG]model_9_1stlay_30000_42minibat_16_filt_split13.pkl', model)
save_volumes('/home/joao/Code/ModelsVolumes/[TRACKING+3D+AUG]volumes_9_1stlay_30000_42minibat_16_filt_split13.pkl', predicted_results)

print(predicted_results.shape)
labeled_results=augmented_val_volumes
print(labeled_results.shape)

index = {0: 'CB', 1: 'mWM', 2: 'BGT', 3: 'vCSF', 4: 'uWM', 5: 'BS', 6: 'GM', 7: 'eCSF'}
correlation, correlation_r2, mse=[],[],[]
for r in xrange(8):
    plt.figure()
    plt.plot(predicted_results[:,r:r+1], labeled_results[:,r:r+1], '*')
    plt.xlabel('Test Results')
    plt.ylabel('Real Results')
    plt.plot([0,1],[0,1])
    plt.savefig('3D_54_graph_'+index[r])
    plt.figure()
    mean_1=(labeled_results[:,r:r+1]+predicted_results[:,r:r+1])/2
    diff=labeled_results[:,r:r+1]-predicted_results[:,r:r+1]
    mean_all=np.mean(diff)
    std= np.std(diff, axis=0)
    plt.plot(mean_1, diff, '*')
    plt.xlabel('Mean')
    plt.ylabel('Difference')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axhline(mean_all, color='green', linestyle='--')
    plt.axhline(mean_all + 1.96*std, color='red', linestyle='--')
    plt.axhline(mean_all - 1.96*std, color='red', linestyle='--')
    plt.savefig('3D_54_graph_Bland-Aldman'+index[r])
    correlation.append(np.corrcoef(labeled_results[:,r],predicted_results[:,r]))
    correlation_r2.append(r2_score(labeled_results[:,r],predicted_results[:,r]))
    mse.append(metrics.mean_squared_error(labeled_results[:,r],predicted_results[:,r]))

print("Correlation with CorrCoef for CB, mWM, BGT, vCSF, uWM, BS, GM, eCSF:")
print(correlation[0][0][1], correlation[1][0][1], correlation[2][0][1], correlation[3][0][1], 
      correlation[4][0][1], correlation[5][0][1], correlation[6][0][1], correlation[7][0][1])

print("Correlation Squared for CB, mWM, BGT, vCSF, uWM, BS, GM, eCSF:")
print(correlation[0][0][1]**2, correlation[1][0][1]**2, correlation[2][0][1]**2, correlation[3][0][1]**2, 
      correlation[4][0][1]**2, correlation[5][0][1]**2, correlation[6][0][1]**2, correlation[7][0][1]**2)

print("Correlation R2 for CB, mWM, BGT, vCSF, uWM, BS, GM, eCSF:")
print(correlation_r2[0], correlation_r2[1], correlation_r2[2], correlation_r2[3], 
      correlation_r2[4], correlation_r2[5], correlation_r2[6], correlation_r2[7])

print("MSE for CB, mWM, BGT, vCSF, uWM, BS, GM, eCSF:")
print(mse[0], mse[1], mse[2], mse[3], mse[4], mse[5], mse[6], mse[7])


from math import sqrt
print("RMSE for CB, mWM, BGT, vCSF, uWM, BS, GM, eCSF:")
print(sqrt(mse[0]), sqrt(mse[1]), sqrt(mse[2]), sqrt(mse[3]), 
      sqrt(mse[4]), sqrt(mse[5]), sqrt(mse[6]), sqrt(mse[7]))

print("Whole MSE:")
print(metrics.mean_squared_error(labeled_results,predicted_results))

rms = sqrt(metrics.mean_squared_error(labeled_results,predicted_results))

print("Whole RMSE:")
print(rms)