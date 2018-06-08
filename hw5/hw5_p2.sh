#!./bin/bash
wget 'https://www.dropbox.com/s/i0zvr6e36phn8om/lstm_classifier_bs1.hdf5'
python3 task2_test.py --val $1 --vl $2 --out $3