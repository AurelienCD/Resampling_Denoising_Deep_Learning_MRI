import tensorflow as tf
import tensorflow_io as tfio
from matplotlib import pyplot as plt
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
from skimage import io
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img
import pathlib
from drive.MyDrive.utils.models import get_unet

def process_path(file_path):
    image_bytes=tf.io.read_file(file_path)
    image=tfio.image.decode_dicom_image(image_bytes, scale='preserve', dtype=tf.uint16)
    return image[0,:,:,:] 

def casting(input_image):
    input_image = tf.cast(input_image, tf.float32)
    return input_image
    
def rotating_to_original(input_image):
    return tf.image.rot90(input_image, k=3)

def normalizing(input_image):
    return input_image/(2**16)

#verify if add_noise works on 3d nii or only 2d dcm
def noisy_input(input, Sigma = 25):
    input = tf.image.resize(input, [256, 256], method="bilinear")
    br_reel = tf.random.normal([256,256,1], 0,Sigma)
    br_imag = tf.random.normal([256,256,1], 0,Sigma)
    tnsr_noisy = tf.complex(input+br_reel,br_imag)
    image_bruitee = tf.math.abs(tnsr_noisy)
    return image_bruitee

def blurry_input(input):
    temporary = tf.image.resize(input, [128, 128], method="bilinear")
    return tf.image.resize(temporary, [256, 256], method="bilinear")

def process_input(input, task):
    if task == 'resampling':
        return blurry_input(input)
    elif task == 'denoising':
        return noisy_input(input)
    else :
        print ("The argument task should be either the string : \"denoising\", or \"resampling\". Try again with a valid argument.")


def process_target(input):
    return tf.image.resize(input, [256, 256], method="bicubic")

def prepare_data(directory, task):
    ds_train = tf.data.Dataset.list_files(str(pathlib.Path(directory+'train/'+'*.dcm')))
    ds_valid = tf.data.Dataset.list_files(str(pathlib.Path(directory+'val/'+'*.dcm')))
    ds_train=ds_train.map(process_path)
    ds_valid=ds_valid.map(process_path)
    ds_train=ds_train.map(casting)
    ds_valid=ds_valid.map(casting)
    ds_train=ds_train.map(rotating_to_original)
    ds_valid=ds_valid.map(rotating_to_original)
    #ds_train=ds_train.map(normalizing)
    #ds_valid=ds_valid.map(normalizing)
    ds_train = ds_train.map(lambda x: (process_input(x, task), process_target(x)))
    ds_train = ds_train.prefetch(buffer_size=32)
    ds_valid = ds_valid.map(lambda x: (process_input(x, task), process_target(x)))
    ds_valid = ds_valid.prefetch(buffer_size=32)
    ds_train=ds_train.batch(8)
    ds_valid=ds_valid.batch(8)
    print("A visualization of the first training batch :")
    for batch in ds_train.take(1):
        for img in batch[0]:
            display(array_to_img(img))
        for img in batch[1]:
            display(array_to_img(img))
    return ds_train, ds_valid

#### For evaluation on test data

def test_nii(file):
    reader = sitk.ReadImage(file)
    reader = sitk.Cast(reader, sitk.sitkFloat32)
    tensor = sitk.GetArrayFromImage(reader)
    tensor = np.swapaxes(tensor,0,2)
    # pour i = 0
    resize=cv2.resize(tensor[:,:,0], dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    #resize = add_np_noise(resize)
    pred=unet.predict(resize[np.newaxis,:,:,np.newaxis])
    temp=np.squeeze(pred)
    temp=temp[:,:,np.newaxis]
    #pred=np.squeeze(pred)
    for i in range (1,tensor.shape[2]):        
        
        resize=cv2.resize(tensor[:,:,i], dsize=(256,256), interpolation=cv2.INTER_CUBIC)
        #resize = add_np_noise(resize)
        pred=unet.predict(resize[np.newaxis,:,:,np.newaxis])
        pred=np.squeeze(pred)
        temp = np.concatenate((temp[:,:,:],pred[:,:,np.newaxis]), axis=2)

    temp = np.swapaxes(temp,0,2)
    return temp

def test_tif(file):
    im = io.imread(file)
    tmp = np.swapaxes(im,0,2)
    tensor = np.swapaxes(tmp,0,1)
    # pour i = 0
    resize=cv2.resize(tensor[:,:,0], dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    #resize = add_np_noise(resize)
    pred=unet.predict(resize[np.newaxis,:,:,np.newaxis])
    temp=np.squeeze(pred)
    temp=temp[:,:,np.newaxis]
    #pred=np.squeeze(pred)
    for i in range (1,tensor.shape[2]):        
        
        resize=cv2.resize(tensor[:,:,i], dsize=(256,256), interpolation=cv2.INTER_CUBIC)
        #resize = add_np_noise(resize)
        pred=unet.predict(resize[np.newaxis,:,:,np.newaxis])
        pred=np.squeeze(pred)
        temp = np.concatenate((temp[:,:,:],pred[:,:,np.newaxis]), axis=2)

    return temp

#def test_and_save(file):
def test_tif(file, unet):
    im = io.imread(file)
    tmp = np.swapaxes(im,0,2)
    tensor = np.swapaxes(tmp,0,1)
    # pour i = 0
    resize=cv2.resize(tensor[:,:,0], dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    #resize = add_np_noise(resize)
    pred=unet.predict(resize[np.newaxis,:,:,np.newaxis])
    temp=np.squeeze(pred)
    temp=temp[:,:,np.newaxis]
    #pred=np.squeeze(pred)
    for i in range (1,tensor.shape[2]):        
        
        resize=cv2.resize(tensor[:,:,i], dsize=(256,256), interpolation=cv2.INTER_CUBIC)
        #resize = add_np_noise(resize)
        pred=unet.predict(resize[np.newaxis,:,:,np.newaxis])
        pred=np.squeeze(pred)
        temp = np.concatenate((temp[:,:,:],pred[:,:,np.newaxis]), axis=2)

    return temp

def tif_to_np(file):
    im = io.imread(file)
    #swapaxes 0 and 2 (and then 0 and 1 idk)
    tmp = np.swapaxes(im,0,2)
    img = np.swapaxes(tmp,0,1)
    return img

def visualize(input): #this function visualize interactively all slices
    def explore_3dimage(layer):
        plt.figure(figsize=(15, 10))
        plt.imshow(input[:, :, layer], cmap='gray');
        plt.title('Explore Layers of Brain MRI', fontsize=20)
        plt.axis('off')
        return layer
        
    interact(explore_3dimage, layer=(0, input.shape[2] - 1));


