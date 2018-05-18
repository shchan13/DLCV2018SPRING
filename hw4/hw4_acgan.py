import numpy as np
import pandas as pd 
from skimage import io
import matplotlib.pyplot as plt
import glob
import pickle 
from sys import argv

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


class acgan_model():
    def __init__(self,batch_size=32, epochs=200, img_size=64, img_dim=3,latent_dim=256, g_iter=2, seed=689):
        # variables
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size 
        self.img_dim = img_dim
        self.latent_dim = latent_dim 
        self.g_iter = g_iter
        self.seed = seed  

        losses =  ['binary_crossentropy', 'binary_crossentropy']
        opt = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
        
        # Build and compile discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile( loss = losses,
                                    optimizer = opt,
                                    metrics=['accuracy'])

        self.discriminator.summary()
        print('\n')

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()
        print('\n')

        # Generator take noise as input
        z = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([z, label])

        # For the combined model we will train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid, target_label = self.discriminator(img)

        # Build and compile combined model
        self.combined = Model([z, label], [valid, target_label])
        self.combined.compile(  loss = losses,
                                optimizer = opt,
                                metrics = ['accuracy'])
        print('combined done')
        self.combined.summary()
        return 


    def build_generator(self):
        print('build generator')

        latent_input = Input(shape=(self.latent_dim,), name='latent_input')
        label_input = Input(shape=(1,), name='label_input')
        model_input = Concatenate()([latent_input, label_input])

        x = Dense(units=8*8*128, activation='relu')(model_input)
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

        img_output = Conv2D(self.img_dim, kernel_size=(3,3), strides=(1,1), activation='tanh', 
                            padding='same', name='generator_conv4')(x)
        
        return Model([latent_input, label_input], img_output)

    
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
        
        feature = Flatten()(x)

        valid_output = Dense(units=1, activation='sigmoid', name = 'valid_output')(feature)
        label_output = Dense(units=1, activation='sigmoid', name='label_output')(feature)

        return Model(img_input, [valid_output, label_output])


    def load_img(self, train_path, train_idx):
        # the size of train_idx = half_batch
        def normalize(img):
            return (img.astype(np.float32) - 127.5) / 127.5

        x_train = np.zeros((train_idx.shape[0], self.img_size, self.img_size, self.img_dim))
        for i in range(train_idx.shape[0]):
            x_train[i,:,:,:] = normalize(io.imread(train_path+format(train_idx[i],'05d')+'.png'))
        
        return x_train 


    def load_label(self, label_path, train_idx):
        return pd.read_csv(label_path+'train.csv')['Smiling'][train_idx].values



    def train(self,smooth_flag,train_path,label_path,model_path,history_path,out_img_path):
        
        print('\nsmooth flag = ', smooth_flag)

        train_num = len(glob.glob1(train_path, '*.png'))
        print('train_num = ', train_num)

        half_batch = self.batch_size // 2
        loop_count = train_num // half_batch 
        
        d_loss_list = [] # Total loss od discriminator
        d_accu_list = [] # real/fake accuracy of discriminator
        d_attr_list = [] # attribute accuracy of discriminator

        g_loss_list = []
        g_accu_list = []
        g_attr_list = [] # attribute accuracy of generator

        for epoch in range(self.epochs):

            # Random shuffle total real_img
            idx = np.random.permutation(train_num)
            # Train Discriminator on batch
            if smooth_flag == True:
                # Add label smoothing: real=0.8~1.0 / fake=0.0~0.2
                real_label=(1.0-0.8)*np.random.normal(loc=1.0,scale=0.2,size=(half_batch,1))+0.8 
                fake_label=(0.2-0.0)*np.random.normal(loc=0.2, scale=0.2, size=(half_batch, 1))

            else:
                real_label = np.ones((half_batch, 1))
                fake_label = np.zeros((half_batch, 1))
            

            # Generator wants Discriminator to label fake imgs as 1
            valid_y = np.ones((self.batch_size,1))


            for step in range(loop_count):
                s_idx = step * half_batch 
                e_idx = (step+1) * half_batch 
               

                # load real images and labels
                x_train = self.load_img(train_path, idx[s_idx:e_idx,])
                x_label = self.load_label(label_path, idx[s_idx:e_idx,])

                # Generate images from Generator given noisy input
                noise = np.random.normal(loc=0.0, scale=1.0, size=(half_batch, self.latent_dim))
                gen_label = np.random.randint(0,2, (half_batch, 1)) # Random sampling the label
                
                gen_imgs = self.generator.predict([noise, gen_label])

                # Train discriminator
                # dr: discriminator real, df: discriminator fake
                dr_loss=self.discriminator.train_on_batch(x_train, [real_label, x_label])
                df_loss=self.discriminator.train_on_batch(gen_imgs, [fake_label, gen_label])
                d_loss = 0.5 * np.add(dr_loss, df_loss)
                
                # Train generator
                noise=np.random.normal(loc=0.0,scale=1.0,size=(self.batch_size, self.latent_dim))
                gen_label = np.random.randint(0,2, (self.batch_size, 1))

                g_loss = self.combined.train_on_batch([noise, gen_label], [valid_y, gen_label])

                # Plot the progress
                print('epoch: %3d/%3d step: %5d/%5d  [D loss: %.4f, acc: %.4f, attr_acc: %.4f] [G loss: %.4f]' % (epoch, self.epochs, step, loop_count, d_loss[0], d_loss[3], d_loss[4], g_loss[0]), end='\r')
                
                d_loss_list.append(d_loss[0])
                d_accu_list.append(d_loss[3])
                d_attr_list.append(d_loss[4])

                g_loss_list.append(g_loss[0])
                g_accu_list.append(g_loss[3])
                g_attr_list.append(g_loss[4])


            with open(history_path+'d_loss.pickle', 'wb') as f:
                pickle.dump(d_loss_list, f)

            with open(history_path+'d_accu.pickle', 'wb') as f:
                pickle.dump(d_accu_list, f)

            with open(history_path+'d_attr.pickle', 'wb') as f:
                pickle.dump(d_attr_list, f)

            with open(history_path+'g_loss.pickle', 'wb') as f:
                pickle.dump(g_loss_list, f)

            with open(history_path+'g_accu.pickle', 'wb') as f:
                pickle.dump(g_accu_list, f)
            
            with open(history_path+'g_attr.pickle', 'wb') as f:
                pickle.dump(g_attr_list, f)
 
                
            # Save generated image samples
            self.sample_images(epoch, out_img_path)
            
            # Save model and history
            self.discriminator.save(model_path+'discriminator_'+str(epoch)+'.hdf5')
            self.generator.save(model_path+'generator_'+str(epoch)+'.hdf5')
        
        return 

    def sample_images(self, epoch, out_img_path):
        
        r,c = 2,10
        np.random.seed(self.seed)
        noise = np.random.normal(0,1, (c, self.latent_dim))
        noise = np.vstack((noise, noise))

        gen_label = np.vstack(( np.zeros((r*c//2,1)), np.ones((r*c//2,1)) ))

        gen_imgs = self.generator.predict([noise,gen_label])

        # Rescale images
        gen_imgs = ((gen_imgs*127.5) + 127.5).astype(np.uint8)

        fig, axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(out_img_path+'celebA_{}.png'.format(epoch))
        plt.close()
        return


def main():
    DATA_PATH = argv[1]

    model = acgan_model()
    model.train(smooth_flag = False,
                train_path = DATA_PATH+'train/',
                label_path = DATA_PATH,
                model_path = './model/',
                history_path = './history/',
                out_img_path = './epoch_gan/')
    print('\nDone!')

    # load model and sample images per epoch
    for epoch in range(model.epochs):
        model.discriminator = load_model('./model/discriminator_{}.hdf5', epoch)
        model.sample_images(epoch, './load_model_sample/')

    return 


if __name__ == '__main__':
    main()
