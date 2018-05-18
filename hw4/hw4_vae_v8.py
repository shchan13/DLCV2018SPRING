import numpy as np
import pandas as pd 
from skimage import io
import matplotlib.pyplot as plt
import glob 
import pickle

from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D, Dropout, Conv2DTranspose, Lambda, Cropping2D, Reshape, Flatten, Dense
from keras.losses import mse
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model 
from keras import backend as K

class vae_model():
    def __init__(self,bs=16,epochs=120,img_size=64,img_dim=3,latent_dim=256,kl_lambda=2e-5):
        # variables
        self.batch_size = bs
        self.epochs = epochs
        self.img_size = img_size 
        self.img_dim = img_dim
        self.latent_dim = latent_dim 
        self.kl_lambda = kl_lambda 

        # output model
        self.encoder = None
        self.decoder = None
        self.vae = None 
        return


    def sample(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        print('batch = ', batch)
        epsilon = K.random_normal(shape = (batch, dim))
        return z_mean + K.exp(0.5*z_log_var)*epsilon 


    def build_model(self):
        # Build encoder
        img_input = Input(shape = (self.img_size, self.img_size, self.img_dim)) # 64,64
        x = Conv2D(8, (3,3), activation='relu', padding='same')(img_input)
        x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3,3), activation='relu', padding='same')(x)

        shape = K.int_shape(x)
        print('shape = ', shape)
        
        # Generate latent vector
        x = Flatten()(x)
        # x = Dense(128, activation='relu')(x)
        z_mean = Dense(self.latent_dim, activation='linear', name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, activation='linear', name='z_log_var')(x)
        z = Lambda(self.sample, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # Encoder model 
        self.encoder = Model(inputs=img_input, outputs=[z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        plot_model(self.encoder, to_file='vae_encoder.png', show_shapes=True)

        # Build decoder
        print('Build decoder')
        latent_input = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(latent_input)
        x = Reshape((shape[1],shape[2],shape[3]))(x)
        x = Conv2DTranspose(32, (3,3), activation='relu', strides=1, padding='same')(x)
        x = Conv2DTranspose(16, (3,3), activation='relu', strides=1, padding='same')(x)
        x = Conv2DTranspose(8, (3,3), activation='relu', strides=1, padding='same')(x)
        img_output = Conv2DTranspose(3, (3,3), activation='tanh', padding='same', name='decoder_output')(x)

        self.decoder = Model(inputs=latent_input, outputs=img_output, name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, to_file='vae_decoder.png', show_shapes=True)
        
        # Build autoencoder
        print('Build vae')
        vae_output = self.decoder(self.encoder(img_input)[2])
        self.vae = Model(inputs=img_input, outputs=vae_output, name='vae')
        self.vae.summary()
        plot_model(self.vae, to_file='vae_autoencoder.png', show_shapes=True)

        # Compile model
        print('Compile vae model')

        def vae_loss(y_true, y_pred):
            reconstruction_loss = mse(K.flatten(y_true), K.flatten(y_pred))
            kld_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            vae_loss = K.mean(reconstruction_loss + self.kl_lambda*kld_loss)
            return vae_loss 

        def kld_loss(y_true, y_pred):
            kld_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return kld_loss

        def reconstruction_loss(y_true, y_pred):
            return mse(y_true, y_pred)

        self.vae.compile(   optimizer=Adam(lr=1e-5, beta_1=0.5, beta_2=0.9), 
                            loss = vae_loss, 
                            metrics=[kld_loss, reconstruction_loss])

        return 


    def train(self, model_path, history_path, x_train, x_val):
        checkpoint = ModelCheckpoint(   model_path,
                                        monitor = 'val_loss',
                                        save_best_only = True,
                                        verbose = 1,
                                        save_weights_only = True)
        
        history = self.vae.fit( x_train, x_train,
                                shuffle = True,
                                epochs = self.epochs,
                                batch_size = self.batch_size,
                                validation_data = (x_val, x_val),
                                callbacks = [checkpoint])

        with open(history_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    
        return                
    

def load_data(img_path, img_size, img_dim, train_flag):

    def normalize(img):
        return (img.astype(np.float32) - 127.5) /127.5

    if train_flag == True:
        print('loading training data...')
        img_num = len(glob.glob1(img_path, '*.png'))

        x_train = np.zeros((int(img_num*0.9), img_size, img_size, img_dim))
        x_val = np.zeros(((img_num - int(img_num*0.9)), img_size, img_size, img_dim))

        for i in range(x_train.shape[0]):
            x_train[i,:,:,:] = normalize(io.imread(img_path+format(i,'05d')+'.png'))

        print('loading validation_data')
        for i in range(x_val.shape[0]):
            x_val[i,:,:,:] = normalize(io.imread(img_path+format(i+x_train.shape[0],'05d')+'.png'))

        print('x_train shape = ', x_train.shape)
        print('x_val shape = ', x_val.shape)
       
        np.save('x_train.npy', x_train)
        np.save('x_val.npy', x_val)
        return x_train, x_val

    else:
        print('loading testing data ...')
        img_num = len(glob.glob1(img_path, '*.png'))
        x_test = np.zeros((img_num, img_size, img_size, img_dim))

        for i in range(x_test.shape[0]):
            x_test[i,:,:,:] = normalize(io.imread(img_path+format(i+40000,'05d')+'.png'))

        np.save('x_test.npy', x_test)
        return x_test 


def main():
    IMG_PATH = '../hw4_data/train/'
    TEST_PATH = '../hw4_data/test2/'
    MODEL_PATH = './model/vae_weight_256.hdf5'
    HISTORY_PATH = './history/vae_history_256.pickle'
    PREDICT_PATH = './predict/'
    GEN_PATH = './gen/'

    TRAIN_FLAG = True
    IMG_SIZE = 64
    IMG_DIM = 3
    IMG_NUM = 10
    RANDOM_NUM = 32
    LATENT_DIM = 256

    img_train, img_val = load_data(IMG_PATH, IMG_SIZE, IMG_DIM, TRAIN_FLAG)  

    model = vae_model(latent_dim=LATENT_DIM)
    model.build_model()
    model.train(MODEL_PATH, HISTORY_PATH, img_train, img_val)
    # model.vae.load_weights(MODEL_PATH)

    
    img_test = load_data(TEST_PATH, IMG_SIZE, IMG_DIM, False)
    print('img_test shape = ', img_test.shape)
    # img_test = np.load('../npy/x_test.npy')
    img_pred = model.vae.predict(img_test, verbose=1)
    print('img_pred shape = ', img_pred.shape)

    img_pred = (img_pred*127.5 + 127.5).astype(np.uint8)
    for i in range(img_pred.shape[0]):
        io.imsave(PREDICT_PATH+format(i+40000,'05d')+'.png',img_pred[i,:,:,:])

    print('decoding randomly')
    np.random.seed(100)
    img_code = np.random.normal(size=(RANDOM_NUM,LATENT_DIM))
    print('img_code shape = ', img_code.shape)
    img_gen = model.decoder.predict(img_code, batch_size=8, verbose=1)
    print('img_gen shape = ', img_gen.shape)
    
    img_gen = (img_gen*255.).astype(np.uint8)

    for i in range(img_gen.shape[0]):
        io.imsave(GEN_PATH+format(i,'05d')+'.png', img_gen[i,:,:,:])

    """
    print('encoder prediction')
    tmp_latent = model.encoder.predict(img_val, batch_size=8, verbose=1)
    print('len tmp_latent = ', len(tmp_latent))
    img_latent = tmp_latent[0]
    print('img_latent shape = ', img_latent.shape)
    print('max img_latent = ', np.max(img_latent))
    print('img_latent[0] = ')
    np.set_printoptions(threshold=np.inf)
    print(img_latent[0,:])
    """
    return 


if __name__ == '__main__':
    main()