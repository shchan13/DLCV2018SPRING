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
#### Task2:
```shell
$ bash task2.sh $1 $2
```
* **$1:** Directory of testing images
* **$2:** Directory of output label label folder

### To train our model
1. Put the folder **"dlcv_final_2_dataset"** under this directory.
2. Run `$ python3 -O ./src/train_mynet.py`.
3. For data augmentation, run `$ python3 -O ./src/train_new.py`.
4. The reult model and history will be saved in *./model* and *./history* respectively.
