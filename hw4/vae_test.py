import numpy as np
import pandas as pd 
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import colors
import glob 
import pickle
from sys import argv
from sklearn.manifold import TSNE


from keras.models import Model, load_model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization 
from keras.metrics import binary_accuracy, mae 
from keras.losses import mse
from keras.optimizers import Adam, SGD, RMSprop 
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model 
from keras import backend as K


class vae_model():
    def __init__(self,bs=16,epochs=120,img_size=64,img_dim=3,latent_dim=128,kl_lambda=2e-5):
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
        img_output = Conv2DTranspose(3, (3,3), activation='sigmoid', padding='same', name='decoder_output')(x)

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

def main():
    
    data_path = argv[1] # Get data path
    out_img_path = argv[2] # output path
    img_size = 64
    img_dim = 3
    random_seed = 689


    #################### VAE #########################
    vae_test = vae_model()
    vae_test.build_model()
    vae_test.vae.load_weights('./vae_weight4.hdf5') # wget
    
    print('\nVAE Encoder')
    vae_test.encoder.summary()

    print('\nVAE Decoder')
    vae_test.decoder.summary()
    

    # Plot learning curve (fig1_2)
    vae_history = pickle.load(open('./vae/vae_history4.pickle', 'rb'))

    plt.figure(num=0, figsize=(15,6))
    plt.subplot(121)
    plt.plot(vae_history['kld_loss'])
    plt.title('KLD Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(122)
    plt.title('Reconstruction loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(vae_history['reconstruction_loss'])
    plt.savefig(out_img_path + 'fig1_2.jpg')
    plt.close()

    # Plot 10 random testing images (fig1_3)
    test_num = len(glob.glob1(data_path+'test/', '*.png'))
    print('\ntest_num = ', test_num)
    
    np.random.seed(random_seed)
    test_idx = np.random.permutation(test_num)[:10,]
    print('test_idx = ', test_idx)

    img_test = np.zeros((10, img_size, img_size, img_dim))
    for i in range(10):
        img_test[i,:,:,:] = io.imread(data_path+'test/'+format(40000+test_idx[i],'05d')+'.png')

    img_pred = vae_test.vae.predict(img_test.astype(np.float32)/255.)
    img_pred = (img_pred * 255.).astype(np.uint8)


    fig, axs = plt.subplots(2,10)
    
    for i in range(10):
        axs[0,i].imshow(img_test[i,:,:,:].astype(np.uint8))
        axs[0,i].axis('off')
        axs[1,i].imshow(img_pred[i,:,:,:])
        axs[1,i].axis('off')
    
    fig.savefig(out_img_path + 'fig1_3.jpg')
    plt.close()


    # Evaluate loss of testing images
    test_idx = np.arange(test_num)
    print('test_idx = ', test_idx)

    img_test = np.zeros((test_num, img_size, img_size, img_dim))
    for i in range(test_num):
        img_test[i,:,:,:] = io.imread(data_path+'test/'+format(40000+test_idx[i],'05d')+'.png')

    img_test = img_test.astype(np.float32) / 255.

    test_result = vae_test.vae.evaluate(img_test, img_test, verbose=0)
    print('test_result = ', test_result)

    # Plot 32 random generated images (fig1_4)
    np.random.seed(random_seed)
    noise = np.random.normal(0,1, (32, vae_test.latent_dim))

    img_gen = vae_test.decoder.predict(noise, verbose=1)
    img_gen = (img_gen * 255.).astype(np.uint8)

    fig, axs = plt.subplots(4,8)
    cnt = 0
    for i in range(4):
        for j in range(8):
            axs[i,j].imshow(img_gen[cnt,:,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(out_img_path + 'fig1_4.jpg')
    plt.close()


    # Encoded latent visualize with TSNE (fig1_5)
    latent_dim = vae_test.encoder.predict(img_test)[0].astype(np.float32)
    print('latent_dim shape = ', latent_dim.shape)

    print('TSNE ...')
    np.random.seed(random_seed)
    latent_enbedded = TSNE(n_components=2).fit_transform(latent_dim)
    print('latent_enbedded shape = ', latent_enbedded.shape)

    # load attributes
    attr_test = pd.read_csv(data_path+'test.csv')['Male'].values.astype(np.uint8)
    print('attr_test shape = ', attr_test.shape)
    print(attr_test[:10,])

    male = latent_enbedded[attr_test==1]
    
    female = latent_enbedded[attr_test==0]
    print('male shape = ', male.shape)

    plt.scatter(male[:,0], male[:,1], cmap='b', label='Male', s=2)
    plt.scatter(female[:,0], female[:,1], cmap='r', label='Female', s=2)
    
    plt.legend(loc='upper right')
    plt.title('Visualization')
    plt.savefig(out_img_path+'fig1_5.jpg')
    plt.close()

    print('VAE Done')
    return

if __name__ == '__main__':
    main()