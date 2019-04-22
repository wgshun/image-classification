# 图像分类

### Finetune AlexNet with Tensorflow
原作者：[kratzert](https://github.com/kratzert/finetune_alexnet_with_tensorflow)

运行环境：Python 3 & Tensorflow >= 1.4 & Numpy

支持 Tensorboard 实时监控训练状态。

训练时，数据集结构：
```
Example train.txt:
/path/to/train/image1.png 0
/path/to/train/image2.png 1
/path/to/train/image3.png 2
/path/to/train/image4.png 0
.
.
```
可使用 `gen_train&val_txt.py` 生成 train.txt 和 val.txt.

预训练模型下载路径：[bvlc_alexnet.npy](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy)

