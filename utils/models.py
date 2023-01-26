import tensorflow 

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, Concatenate ,add
from tensorflow.keras.models import Model 
from tensorflow.keras import regularizers

def get_unet():

    #encoder

    input_img = Input(shape=(256,256,1))
    conv1 = Conv2D(64,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(input_img)
    pool1 = MaxPooling2D(padding='same')(conv1)

    conv2 = Conv2D(128,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool1)
    pool2 = MaxPooling2D(padding='same')(conv2)

    conv3 = Conv2D(256,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool2)
    pool3 = MaxPooling2D(padding='same')(conv3)

    conv4 = Conv2D(512,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool3)
    conv4 = Conv2D(512,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(conv4)

    #decoder

    up5 = Conv2DTranspose(256,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv4)

    merge5 = Concatenate(axis = 3)([up5,conv3])
    conv5 = Conv2D(256,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge5)

    up6 = Conv2DTranspose(128,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv5)

    merge6 = Concatenate(axis = 3)([up6,conv2])
    conv6 = Conv2D(128,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge6)

    up7 = Conv2DTranspose(64,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv6)

    merge7 = Concatenate(axis = 3)([up7,conv1])
    conv7 = Conv2D(64,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge7)

    output = Conv2D(1,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(conv7)
    unet = Model(input_img,output)
    unet.summary()
    return unet

def get_unet4():
    # input 128 => output 256

    #UNETSR+ depth 4

    #encoder

    input_img = Input(shape=(128,128,1))

    conv0 = Conv2DTranspose(32,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(input_img)

    conv1 = Conv2D(64,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(input_img)
    pool1 = MaxPooling2D(padding='same')(conv1)

    conv2 = Conv2D(128,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool1)
    pool2 = MaxPooling2D(padding='same')(conv2)

    conv3 = Conv2D(256,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool2)
    pool3 = MaxPooling2D(padding='same')(conv3)

    conv4 = Conv2D(512,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool3)
    pool4 = MaxPooling2D(padding='same')(conv4)
                                        
    conv5 = Conv2D(1024,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool4)
    conv5 = Conv2D(1024,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(conv5)

    #decoder

    up5 = Conv2DTranspose(512,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv5)

    merge5 = Concatenate(axis = 3)([up5,conv4])
    conv5 = Conv2D(512,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge5)

    up6 = Conv2DTranspose(256,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv5)

    merge6 = Concatenate(axis = 3)([up6,conv3])
    conv6 = Conv2D(256,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge6)

    up7 = Conv2DTranspose(128,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv6)

    merge7 = Concatenate(axis = 3)([up7,conv2])
    conv7 = Conv2D(128,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge7)

    up8 = Conv2DTranspose(64,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv7)

    merge8 = Concatenate(axis = 3)([up8,conv1])
    conv8 = Conv2D(64,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge8)

    up9 = Conv2DTranspose(32,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv8)

    merge9 = Concatenate(axis = 3)([up9,conv0])
    conv9 = Conv2D(32,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge9)

    output = Conv2D(1,(3,3),padding='same',activity_regularizer=regularizers.l1(10e-10))(conv9) #softmax as z lu paper

    unet = Model(input_img,output)
    unet.summary()
    return unet

def get_unet3():
    # input 128 => output 256

    #UNETSR+ depth 3

    #encoder

    input_img = Input(shape=(128,128,1))

    conv0 = Conv2DTranspose(32,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(input_img)

    conv1 = Conv2D(64,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(input_img)
    pool1 = MaxPooling2D(padding='same')(conv1)

    conv2 = Conv2D(128,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool1)
    pool2 = MaxPooling2D(padding='same')(conv2)

    conv3 = Conv2D(256,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool2)
    pool3 = MaxPooling2D(padding='same')(conv3)

    conv4 = Conv2D(512,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(pool3)
    conv4 = Conv2D(512,(3,3),padding='same',activation='relu', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l1(10e-10))(conv4)

    #decoder

    up5 = Conv2DTranspose(256,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv4)

    merge5 = Concatenate(axis = 3)([up5,conv3])
    conv5 = Conv2D(256,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge5)

    up6 = Conv2DTranspose(128,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv5)

    merge6 = Concatenate(axis = 3)([up6,conv2])
    conv6 = Conv2D(128,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge6)

    up7 = Conv2DTranspose(64,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv6)

    merge7 = Concatenate(axis = 3)([up7,conv1])
    conv7 = Conv2D(64,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge7)

    up8 = Conv2DTranspose(32,(3,3),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',activity_regularizer=regularizers.l1(10e-10))(conv7)

    merge8 = Concatenate(axis = 3)([up8,conv0])
    conv8 = Conv2D(64,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(merge8)

    output = Conv2D(1,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(conv8) #softmax instead of relu as z lu
    unet = Model(input_img,output)
    unet.summary()
    return unet


