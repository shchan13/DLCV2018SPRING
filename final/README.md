# DLCV Final Project - Challenge 2
### Group11
### Members: 
* Shao-Hung Chan R06921017
* Yu-Ting Hsiao  R06921010
* Ya-Hui Chien   R06921016

## Usage
### Main Dependencies:
1. Python 3.5+
2. numpy 1.14.3
3. pandas 0.23.0
4. scikit-image 0.13.1
5. keras 2.1.5

### Run Testing
#### Task1:
```shell
$ bash task1.sh $1 $2
```
* **$1:** Absolute directory of testing images. eg. <YOUR_PATH>/test/
* **$2:** Absolute directory of output label **folder**. eg. <YOUR_PATH>/result/

The output csv file will be **output_inception_v3_new2.csv**

#### Task2:
```shell
$ bash task2.sh $1 $2
```
* **$1:** Absolute directory of testing images. eg. <YOUR_PATH>/test/
* **$2:** Absolute directory of output label **folder**. eg. <YOUR_PATH>/result/

Since we implemented 4 models, the output files will be **output_xception_aug2.csv**, **output_shufflenet.csv**, **output_mobilenet.csv**, and **output_mynet2_aug.csv** inside $2 respectively.

### To train our model
1. Put the folder **"dlcv_final_2_dataset"** under this directory.
2. Run `$ python3 -O ./src/train_mynet.py`.
3. For data augmentation, run `$ python3 -O ./src/train_new.py`.
4. The reult model and history will be saved in *./model* and *./history* respectively.