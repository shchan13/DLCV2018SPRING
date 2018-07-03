import numpy as np
from skimage import io 
import keras
from keras.preprocessing.image import ImageDataGenerator


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, img_dir, use_aug, 
                 batch_size=8, dim=(218, 178), n_channels=3, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.img_dir = img_dir
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.use_aug = use_aug
        self.datagen = ImageDataGenerator(zoom_range=0.1, rotation_range=0.1, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    def normalize(self, img):
        return img/127.5 - 1.0 # range to [-1,1] 

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.normalize(io.imread(self.img_dir + ID).astype(np.float32))

            # Store class
            y[i] = labels_temp[i]

        if self.use_aug == True:
            (X, y) = next(self.datagen.flow(X, y, batch_size = self.batch_size))

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)