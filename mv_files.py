import os
import shutil
from glob import glob

train_path = './data/public/train'
test_path = './data/public/test'

all_train_img_list = glob(train_path + '/*/*/*JPG')
all_test_img_list = glob(test_path + '/*/*JPG')

for img_path in all_train_img_list:
    shutil.move(img_path, os.path.join(train_path, img_path.split('/')[-1]))


for img_path in all_test_img_list:
    shutil.move(img_path, os.path.join(test_path, img_path.split('/')[-1]))