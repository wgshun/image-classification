"""
AlexNet 模型的实际测试。

@author: wgshun
"""

import tensorflow as tf
from tf_alexnet import AlexNet
from allClasses_name import classes_name

import time

val_file = '1.jpg'
num_classes = 16

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

def parse_function_val(filename):
    """Input parser for samples of the validation/test set."""

    # load and preprocess the image
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

    # RGB -> BGR
    img_bgr = img_centered[:, :, ::-1]

    return img_bgr
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model = AlexNet(x, keep_prob, 16, [], weights_path='model\\checkpoints\\model_epoch8.ckpt')

    # Link variable to model output
    score = model.fc8

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights into the non-trainable layer
        model.load_model(sess)

        # Validate the model on the entire validation set
        print("Start validation")
        t0 = time.time()
        val_data = parse_function_val(val_file, )
        input = val_data.eval().reshape([1, 227, 227, 3])
        sco = sess.run(score, feed_dict={x: input, keep_prob: 1.})
        print(time.time()-t0)
        print(tf.argmax(sco, 1).eval())
