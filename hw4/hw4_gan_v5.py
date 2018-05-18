import numpy as np
import pandas as pd 
from skimage import io
import matplotlib.pyplot as plt
import glob
import pickle 
from sys import argv

from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization 
from keras.metrics import binary_accuracy, mae 
from keras.losses import mse
from keras.optimizers import Adam, SGD, RMSprop 
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model 
from keras import backend as K


class gan_model():
    def __init__(self, batch_size=32, epochs=120, img_size=64, img_dim=3, latent_dim=128, g_iter=2):
        # variables
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size 
        self.img_dim = img_dim
        self.latent_dim = latent_dim 
        self.g_iter = g_iter 

        
        # Build and compile discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile( loss = 'binary_crossentropy', 
                                    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9),
                                    metrics=['accuracy'])

        self.discriminator.summary()
        print('\n')

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()
        print('\n')

        # Generator take noise as input
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # Build and compile combined model
        self.combined = Model(z, valid)
        self.combined.compile(  loss = 'binary_crossentropy', 
                                optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9),
                                metrics = ['accuracy'])
        print('combined done')
        self.combined.summary()
        return 


    def build_generator(self):
        print('build generator')
        latent_input = Input(shape=(self.latent_dim,), name='latent_input')
        x = Dense(units=8*8*128, activation='relu')(latent_input)
        x = BatchNormalization(momentum=0.8)(x)
        
        x = Reshape((8, 8, 128))(x)
        x = UpSampling2D()(x)
        x = Conv2D( 128, kernel_size=(5, 5), strides=(1, 1), activation='relu', 
                    padding='same', name='generator_conv1')(x)
        x = BatchNormalization(momentum=0.8)(x)
        
        x = UpSampling2D()(x)
        x = Conv2D( 64, kernel_size=(5, 5), strides=(1, 1), activation='relu', 
                    padding='same', name='generator_conv2')(x)
        x = BatchNormalization(momentum=0.8)(x)
        
        x = UpSampling2D()(x)
        x = Conv2D( 64, kernel_size=(5, 5), strides=(1, 1), activation='relu', 
                    padding='same', name='generator_conv3')(x)
        x = BatchNormalization(momentum=0.8)(x)

        img_output = Conv2D(3, kernel_size=(3,3), strides=(1,1), activation='tanh', 
                            padding='same', name='generator_conv4')(x)
        
        return Model(latent_input, img_output)

    
    def build_discriminator(self):
        print('build discriminator')
        img_input = Input(shape = (self.img_size, self.img_size, self.img_dim), name='img_input')

        x = Conv2D(64, (5,5), strides=(2,2), padding='same')(img_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(64, (5,5), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(128, (5,5), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dropout(0.25)(x)
        
        x = Flatten()(x)
        valid_output = Dense(units=1, activation='sigmoid')(x)

        return Model(img_input, valid_output)


    def train(self, x_train, sample_interval=100, smooth_flag=False, model_path='./model/'):
        
        print('\nsmooth flag = ', smooth_flag)
        train_num = x_train.shape[0] # total number of training data
        half_batch = self.batch_size // 2
        loop_count = train_num // half_batch 
        
        d_loss_list = []
        d_accu_list = []
        g_loss_list = []
        g_accu_list = []

        for epoch in range(self.epochs):

            # Random shuffle total real_img
            idx = np.random.permutation(train_num)
            x_train = x_train[idx]
            # print('x_train shape = ', x_train.shape)

            # Train Discriminator on batch
            if smooth_flag == True:
                # Add label smoothing: real=0.8~1.0 / fake=0.0~0.2
                real_label=(1.0-0.8)*np.random.normal(loc=0.8,scale=0.2,size=(half_batch,1))+0.8 
                fake_label=(0.2-0.0)*np.random.normal(loc=0.2, scale=0.2, size=(half_batch, 1))

            else:
                real_label = np.ones((half_batch, 1))
                fake_label = np.zeros((half_batch, 1))
            

            for step in range(loop_count):
                s_idx = step * half_batch 
                e_idx = (step+1) * half_batch 
                # print('s_idx, e_idx = ', s_idx, e_idx)
                
                # Generate images from Generator given noisy input
                # print('Generate noise')
                noise = np.random.normal(loc=0.0, scale=1.0, size=(half_batch, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # print('Train Discriminator')
                # dr: discriminator real, df: discriminator fake
                dr_loss=self.discriminator.train_on_batch(x_train[s_idx:e_idx,:,:,:],real_label)
                df_loss=self.discriminator.train_on_batch(gen_imgs,fake_label)
                d_loss = 0.5 * np.add(dr_loss, df_loss)
                
                
                # Generator wants Discriminator to label fake imgs as 1
                valid_y = np.ones((self.batch_size,1))

                # print('Train Generator')
                
                # Train generator
                noise=np.random.normal(loc=0.0,scale=1.0,size=(self.batch_size, self.latent_dim))
                g_loss = self.combined.train_on_batch(noise, valid_y)

                # Plot the progress
                print('epoch: %3d  step: %3d / %3d' % (epoch, step, loop_count))
                print('D loss: {}, real: {}, fake: {}'.format(d_loss[0], dr_loss[0], df_loss[0]))
                print('D acc : {}, real: {}, fake: {}'.format(d_loss[1], dr_loss[1], df_loss[1]))
                print('G loss: {}, acc: {}'.format(g_loss[0], g_loss[1])) 
                print('-------------------------------------------------------------------------------')

                d_loss_list.append(d_loss[0])
                d_accu_list.append(d_loss[1])
                g_loss_list.append(g_loss[0])
                g_accu_list.append(g_loss[1])

            with open('./d_loss.pickle', 'wb') as f:
                pickle.dump(d_loss_list, f)

            with open('./d_accu.pickle', 'wb') as f:
                pickle.dump(d_accu_list, f)

            with open('./g_loss.pickle', 'wb') as f:
                pickle.dump(g_loss_list, f)

            with open('./g_accu.pickle', 'wb') as f:
                pickle.dump(g_accu_list, f)
                
                
            # Save generated image samples
            self.sample_images(epoch)
            
        
            # Save model and history
            print('save model')
            self.discriminator.save(argv[2] + 'discriminator_{}.hdf5'.format(epoch))
            self.generator.save(argv[2] + 'generator_{}.hdf5'.format(epoch))
        
        return 

    def sample_images(self, epoch):
        
        print('sample images')
        r,c = 4,8
        np.random.seed(1000)
        noise = np.random.normal(0,1, (r*c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        print('gen_imgs shape = ', gen_imgs.shape)

        # Rescale images
        gen_imgs = ((gen_imgs*127.5) + 127.5).astype(np.uint8)
        print('max gen_imgs = ', np.max(gen_imgs))
        print('min gen_imgs = ', np.min(gen_imgs))

        fig, axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('./celebA_{}.png'.format(epoch))
        plt.close()
        return



def load_data(train_path, test_path, img_size=64, img_dim=3, train_flag=True):
    
    def normalize(img):
        return ((img-127.5) / 127.5).astype(np.float32)

    img_num = len(glob.glob1(train_path, '*.png'))
    
    if train_flag == True:
        print('loading training data...')
        print('train img number = ', img_num)
        x_train = np.zeros((img_num, img_size, img_size, img_dim))

        for i in range(img_num):
            x_train[i,:,:,:] = normalize(io.imread(train_path+format(i,'05d')+'.png'))

        print('x_train shape = ', x_train.shape)
        
        return x_train

    else:
        print('loading testing data ...')
        print('test img number = ', img_num)
        x_test = np.zeros((img_num, img_size, img_size, img_dim))

        for i in range(x_test.shape[0]):
            x_test[i,:,:,:] = normalize(io.imread(test_path+format(i+40000,'05d')+'.png')) 

        return x_test 


def main():
    DATA_PATH = argv[1]
    OUTPUT_PATH = argv[2]
    IMG_PATH = DATA_PATH + 'train/'
    TEST_PATH = DATA_PATH + 'test/'


    SMOOTH_FLAG = True 
    IMG_SIZE = 64
    IMG_DIM = 3
    IMG_NUM = 10
    RANDOM_NUM = 32
    LATENT_DIM = 128

    img_train = load_data(train_path=IMG_PATH, test_path=TEST_PATH)
    print('img_train max= {}, min= {}'.format(np.max(img_train), np.min(img_train)))

    model = gan_model()
    model.train(x_train=img_train)
    
    return 


if __name__ == '__main__':
    main()
