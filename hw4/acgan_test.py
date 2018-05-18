import numpy as np
import pandas as pd 
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import spline
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

def main():

    data_path = argv[1] # Get data path
    out_img_path = argv[2] # output path
    img_size = 64
    img_dim = 3
    random_seed = 300
    latent_dim = 256

	# Plot learning curve (fig3_2)
    d_accu = np.asarray(pickle.load(open('./acgan/d_accu.pickle', 'rb')))
    d_loss = np.asarray(pickle.load(open('./acgan/d_loss.pickle', 'rb')))
    d_attr = np.asarray(pickle.load(open('./acgan/d_attr.pickle', 'rb')))
    
    g_accu = np.asarray(pickle.load(open('./acgan/g_accu.pickle', 'rb')))
    g_loss = np.asarray(pickle.load(open('./acgan/g_loss.pickle', 'rb')))
    g_attr = np.asarray(pickle.load(open('./acgan/g_attr.pickle', 'rb')))

    
    x = np.arange(0, d_accu.shape[0], 1)
   
    xsub = np.arange(0, d_accu.shape[0], 1000)
    d_accu_sub = d_accu[xsub]
    g_accu_sub = g_accu[xsub]
    d_loss_sub = d_loss[xsub]
    g_loss_sub = g_loss[xsub]
    d_attr_sub = d_attr[xsub]
    g_attr_sub = g_attr[xsub]


    xnew = np.linspace(xsub.min(), xsub.max(), 100)
    d_loss_s = spline(xsub, d_loss_sub, xnew)
    g_loss_s = spline(xsub, g_loss_sub, xnew)


    plt.figure(num=0, figsize=(28,6))
    plt.subplot(131)
    plt.plot(xnew, d_loss_s)
    plt.plot(xnew, g_loss_s)
    plt.legend(['discriminator', 'generator'])
    plt.title('Total Loss')
    plt.xlabel('steps')
    
    plt.subplot(132)
    plt.plot(xsub, d_attr_sub)
    plt.plot(xsub, g_attr_sub)
    plt.legend(['discriminator', 'generator'])
    plt.title('Attribute Accuracy')
    plt.xlabel('steps')


    plt.subplot(133)
    plt.plot(xsub, d_accu_sub)
    plt.plot(xsub, g_accu_sub)
    plt.legend(['discriminator', 'generator'])
    plt.title('Valid Accuracy')
    plt.xlabel('steps')
    plt.savefig(out_img_path + 'fig3_2.jpg')


	# Load model
    generator = load_model('./acgan/generator_194.hdf5')
   
    # Generate images (fig3_3)
    r,c = 2,10
    np.random.seed(random_seed)
    noise = np.random.normal(0,1, (c, latent_dim))
    noise = np.vstack((noise, noise))

    gen_label = np.vstack(( np.zeros((r*c//2,1)), np.ones((r*c//2,1)) ))

    gen_imgs = generator.predict([noise,gen_label])

    # Rescale images
    gen_imgs = ((gen_imgs*127.5) + 127.5).astype(np.uint8)

    fig, axs = plt.subplots(r,c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(out_img_path + 'fig3_3.jpg')
    plt.close()

    print('\nACGAN Done')

    return

if __name__ == '__main__':
	main()