#!./bin/bash
wget 'https://www.dropbox.com/s/1g75rpa5mj160yn/vae_weight4.hdf5'
python3 vae_test.py $1 $2
python3 gan_test.py $1 $2
python3 acgan_test.py $1 $2
