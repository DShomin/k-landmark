mkdir data
mv ~/Download/data.zip ./data

cd data

unzip data.zip

mkdir train
mkdir test

mv train.zip ./train
mv test.zip ./test

unzip train/train.zip
unzip test/test.zip