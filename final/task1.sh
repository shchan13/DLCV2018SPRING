#!./bin/bash
wget 'https://www.dropbox.com/s/nobwuwyx4k660x0/inception_v3_new2.hdf5?dl=1' -O './model/inception_v3_new2.hdf5'
cd src
python3 task1_test.py --test $1 --out $2