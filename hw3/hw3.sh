#!./bin/bash
wget 'https://www.dropbox.com/s/ymxqg9pnq6kk8v6/vgg16fcn32_weight_azure.hdf5?dl=1'
python3 hw3_test.py $1 $2 0
