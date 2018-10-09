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
    
    return images, volumeperslice, brain_masks, patients

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
    
trainimages, trainvolumeperslice,trainmasks,trainpatients = loadImages(trainimagepaths,segmentationfolder,mask_folder,largestdims, nroflabels)
valimages, valvolumeperslice,valmasks,valpatients = loadImages(valimagepaths, segmentationfolder,mask_folder,largestdims, nroflabels)
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
            center=(min_w+((max_w-min_w)/2), min_h+((max_h-min_h)/2))
        else:
            center=(192,192)
        return  min_w,max_w,min_h,max_h,center
    
def optimized_values(images,masks):
    values, array_center=[],[]
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
        array_center.append(values[x][4])
    return w_real_min,w_real_max,h_real_min,h_real_max,array_center   

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
        w_min, w_max, h_min, h_max=(int(int(values[z][0])-(width/2)),int(int(values[z][0])+(width/2)),int(int(values[z][1])-(height/2)),int(int(values[z][1])+(height/2)))
        if (h_min<0):
            h_max=h_max-h_min
            h_min=0
        reduced_pic.append(images[z,0,h_min:h_max,w_min:w_max])
    np_reduced_pic=np.array(reduced_pic)
    print(np_reduced_pic.shape)
    return np_reduced_pic
 
width=max_width-min_width
height=max_height-min_height
print(width, height)

def CheckDims (w, h):
    if (w % 2)!=0:
        w=w+1
    if (h % 2)!=0:
        h=h+1
    return w,h

width,height=CheckDims(width,height)
    
reduced_trainimages=reduced_pic(trainimages,train_values[4])
reduced_valimages=reduced_pic(valimages, train_values[4])

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

print("------------------------------CNN code-------------------------------")
import lasagne
import theano
from theano import tensor as T
#from lasagne.layers import dnn

def getmodel(X1,p_drop_hidden):     
    convnetwork = lasagne.layers.InputLayer(shape=(None, 1, height, width),input_var=X1)
    for i in xrange(3): 
        convnetwork = lasagne.layers.Conv2DLayer(convnetwork, num_filters=16, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    print convnetwork.output_shape
        
    convnetwork = lasagne.layers.MaxPool2DLayer(convnetwork,pool_size=(2,2))
    print convnetwork.output_shape
    
    for i in xrange(3): 
        convnetwork = lasagne.layers.Conv2DLayer(convnetwork, num_filters=16, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())   
    print convnetwork.output_shape

    convnetwork = lasagne.layers.MaxPool2DLayer(convnetwork,pool_size=(2,2))    
    print convnetwork.output_shape
        
    for i in xrange(3): 
        convnetwork = lasagne.layers.Conv2DLayer(convnetwork, num_filters=16, filter_size=(3, 3),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())   
    print convnetwork.output_shape
        
    convnetwork = lasagne.layers.MaxPool2DLayer(convnetwork,pool_size=(2,2))
    print convnetwork.output_shape
    
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(convnetwork, p=p_drop_hidden),num_units=256,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=p_drop_hidden),num_units=8,nonlinearity=lasagne.nonlinearities.linear)     
    
    return network, convnetwork   
       
X1 = T.tensor4()  
Y = T.matrix()   

model, convmodel = getmodel(X1, 0.5)
py_x_noise = lasagne.layers.get_output(model) 
cost = T.mean(lasagne.objectives.squared_error(py_x_noise, Y))
params = lasagne.layers.get_all_params(model, trainable=True)
updates = lasagne.updates.rmsprop(cost, params, learning_rate=0.001)

train = theano.function(inputs=[X1, Y], outputs=cost, updates=updates, allow_input_downcast=True)

# The functions
outputprediction = lasagne.layers.get_output(model, deterministic=True)
predict = theano.function(inputs=[X1], outputs=outputprediction, allow_input_downcast=True)

convoutput = lasagne.layers.get_output(convmodel, deterministic=True)
get_convoutput = theano.function(inputs=[X1],outputs=convoutput, allow_input_downcast=True)

costvalidation = T.mean(lasagne.objectives.squared_error(outputprediction, Y))
validation = theano.function(inputs=[X1, Y], outputs=costvalidation, allow_input_downcast=True)

def transform2Dimage(image2D,angle,scale,tx,ty): #angle: degrees, scale: factor, tx ty: voxels
    halfsize = image2D.shape[0]/2

    T = transform.AffineTransform(translation=(halfsize,halfsize))
    RS = transform.AffineTransform(rotation=angle/360*2*np.pi,scale=(1/scale,1/scale))
    Tinv = transform.AffineTransform(translation=(-halfsize+tx,-halfsize+ty))

    transfmatrix = np.dot(np.dot(T.params,RS.params),Tinv.params)

    rotatedimage = transform.warp(image2D,transfmatrix)
    
    return rotatedimage

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
train_8_volumes=normalization(trainvolumeperslice,trainvolumeperslice)
val_8_volumes=normalization(valvolumeperslice,trainvolumeperslice)
print(train_8_volumes.shape)
print(val_8_volumes.shape)

def ChangeIntensity(np_image,number,f_number):
    np_new_img=np_image**number
    if f_number==1:
        np_new_img=np.flip(np_new_img,1)
    return np_new_img

def WholeTransformation(batch_images, batch_volumes): 
    transform_matrix=[]
    all_volumes=[]
    for k in (xrange(batch_images.shape[0])):
        scale_factor=random.uniform(0.8, 1)
        change = transform2Dimage(batch_images[k,0],randint(-10,10),scale_factor,randint(-2,2),randint(-14,4))
        i_change=ChangeIntensity(change,np.random.uniform(0.95, 1.05),randint(0,1))
        transform_matrix.append(i_change)
        new_volume=batch_volumes[k]*(scale_factor*scale_factor)
        all_volumes.append(new_volume)
    np_transform_matrix = np.array(transform_matrix)
    np_all_volumes = np.array(all_volumes)
    reshaped_np_transform_matrix=np.reshape(np_transform_matrix,(np_transform_matrix.shape[0],1,np_transform_matrix.shape[1],np_transform_matrix.shape[2]))
    full_batch_images=np.concatenate((batch_images,reshaped_np_transform_matrix),axis=0)
    full_batch_volumes=np.concatenate((batch_volumes,np_all_volumes),axis=0)
    
    return full_batch_images, full_batch_volumes
	
trainingsamples = np.arange(len(trainimages))
validsamples = np.arange(len(valimages))

minibatches = 2
minibatchsize = 150

losslist = []
validlosslist = []
print(train_8_volumes.shape)
print("I am training....")
t0 = time.time()

for i in xrange(minibatches):
    imagesample=random.sample(trainingsamples,minibatchsize)
    t_images=reshaped_np_subsample_train[imagesample]
    t_volumes=train_8_volumes[imagesample]
    full_t_images,full_t_volumes=WholeTransformation(t_images,t_volumes)
    x=train(full_t_images,full_t_volumes)
    losslist.append(x)
    validimagesample=random.sample(validsamples,minibatchsize)
    v_images=reshaped_np_subsample_val[validimagesample]
    v_volumes=val_8_volumes[validimagesample]
    full_v_images,full_v_volumes=WholeTransformation(v_images,v_volumes)
    y=validation(full_v_images,full_v_volumes)
    validlosslist.append(y)

t1 = time.time()
print 'Training time: {} seconds'.format(t1-t0)

plt.figure()
plt.plot(losslist)
plt.plot(validlosslist)
plt.yscale("log")
plt.savefig('2D_validlosslist')

predicted_results=predict(reshaped_np_subsample_val)
labeled_results=val_8_volumes
print(predicted_results.shape)
print(labeled_results.shape)

def save_weights(filename,network):
    with open(filename, 'wb') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(network), f)
def save_volumes(filename, volumes):
    with open(filename, 'wb') as f:
        cPickle.dump(volumes, f)        
        
save_weights('/home/joao/Code/ModelsVolumes/[2D+NewTech]model_2D_1stlay_200000_400minibat_16_filt_split13.pkl', model)
save_volumes('/home/joao/Code/ModelsVolumes/[2D+NewTech]volumes_2D_1stlay_200000_400minibat_16_filt_split13.pkl', predicted_results)

index = {0: 'CB', 1: 'mWM', 2: 'BGT', 3: 'vCSF', 4: 'uWM', 5: 'BS', 6: 'GM', 7: 'eCSF'}
correlation, correlation_r2, mse=[],[],[]
for r in xrange(8):
    plt.figure()
    plt.plot(predicted_results[:,r:r+1], labeled_results[:,r:r+1], '*')
    plt.xlabel('Test Results')
    plt.ylabel('Real Results')
    plt.plot([0,1],[0,1])
    plt.savefig('2D_graph_'+index[r])
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
    plt.savefig('2D_graph_Bland-Aldman_'+index[r])
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

print("Whole MSE:")
print(metrics.mean_squared_error(labeled_results,predicted_results))


