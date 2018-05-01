#!./bin/bash
wget 'https://www.dropbox.com/s/8v4sru5xjvnkoxr/vgg16fcn8_weight_azure2.hdf5?dl=1'
python3 hw3_test.py $1 $2 1
