import pandas as pd
import numpy as np 
import glob
from skimage import io 
import json
import os
import argparse

from keras.models import load_model

# for mobilenet
from keras.applications.mobilenet import *
from keras.utils.generic_utils import CustomObjectScope

def normalize(img):
    return img/127.5 - 1.0 # range to [-1,1] 

def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default=os.getcwd())
    parser.add_argument('--out', type=str, default=os.getcwd())
    args = parser.parse_args()

    MODEL_DIR = ['./model/xception_aug2.hdf5', './model/shufflenet.hdf5', './model/mobilenet.hdf5', './model/mynet2_aug.hdf5']
    TEST_IMG_DIR = args.test
    OUTPUT_DIR = args.out
    OUTPUT_NAME = ['output_xception_aug2.csv', 'output_shufflenet.csv', 'output_mobilenet.csv', 'output_mynet2_aug.csv']

    TEST_NUM = len(glob.glob1(TEST_IMG_DIR, '*.jpg'))
    print('TEST_NUM = ', TEST_NUM)

    IMG_HEIGHT = 218
    IMG_WIDTH = 178
    BATCH_SIZE = 16

    # Load the dictionary for id
    with open('./dictionary/dual_dict.txt', 'r') as f:
        dual_dict = json.load(f)

    with open('./dictionary/id_dict.txt', 'r') as f:
        id_dict = json.load(f)

    N_CLASSES = len(id_dict)

    # Testing
    test_img = np.zeros((TEST_NUM, IMG_HEIGHT, IMG_WIDTH, 3))

    for i in range(TEST_NUM):
        test_img[i,:,:,:] = normalize(io.imread(TEST_IMG_DIR+format(i+1,'05d')+'.jpg').astype(np.float32))
           
    for idx in range(len(MODEL_DIR)):
        # load model
        if idx == 2:
            with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
                model = load_model(MODEL_DIR[idx])
        else:
            model = load_model(MODEL_DIR[idx])
       
        test_pred = model.predict(test_img, verbose=1)
        test_pred = np.argmax(test_pred, axis=-1).astype(int)

        test_pred_dual = [dual_dict[str(ele)] for ele in test_pred]
        test_pred_dual = np.array(test_pred_dual).astype(int).reshape(-1,1)
        print('test_pred_dual shape = ', test_pred_dual.shape)

        # Form output.csv
        id_out = np.array([str(i+1) for i in range(TEST_NUM)]).reshape(-1, 1)
        output = np.hstack((id_out, test_pred_dual))
        output_df = pd.DataFrame(data = output, columns = ['id', 'ans'])
        output_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_NAME[idx]), index = False)
        
        print('Ready for Kaggle !')

        del model

    return


if __name__ == '__main__':
    main()