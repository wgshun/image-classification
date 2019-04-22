"""
根据一定文件夹结构下的图片，生成训练或检验所需的列表文件

Example train.txt:
/path/to/train/image1.png 0
/path/to/train/image2.png 1
/path/to/train/image3.png 2
/path/to/train/image4.png 0
.
.

@author: wgshun
"""

import os

image_path = '..\\train_data'
label_dic = {}
with open(os.path.join(image_path, 'train.txt'), 'w') as txt:
    for root, dirs, files in os.walk(image_path):
        for ind, dir in enumerate(dirs):
            label_dic[dir] = ind
        for file in files:
            if os.path.splitext(file)[-1] != '.txt':
                wr_str = os.path.join(root, file) + ' ' + str(label_dic[root.split('\\')[-1]]) + '\n'
                txt.write(wr_str)
txt.close()