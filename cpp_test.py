# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 10:06  2021-12-02

import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import cv2
import imageio
import numpy as np

def Load_Mnist_Data(savetest=True, num=200):
    """
    load mnsit dataset
    :param savetest:
    :param num:
    :return:
    """
    (xtrain, xtrainlabel), (xtest, xtestlabel) = tf.keras.datasets.mnist.load_data()
    if savetest:
        testdir = "test_images"
        if not os.path.exists(testdir): os.makedirs(testdir)
        for i in range(num):
            testimg = xtest[i]
            testimg = cv2.resize(testimg, (224, 224))
            imageio.imsave(os.path.join(testdir, str(i + 1) + ".png"), testimg)
        xtest, xtestlabel = xtest[:num, ...], xtestlabel[:num]

    xtrainlabel = tf.keras.utils.to_categorical(xtrainlabel, 10)
    xtestlabel = tf.keras.utils.to_categorical(xtestlabel, 10)
    return (xtrain, xtrainlabel), (xtest, xtestlabel)


def Net():
    """
    build model
    :return:
    """
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(224, 224, 3),
                     name='input_image'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())  # # 112 x 112

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())  # # 64 x 64

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())  # # 32 x 32

    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))  # # 16 x 16

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', name='output_class'))

    return model


def PreprocessData(xtrain, xlabel):
    """
    data process
    :param xtrain:
    :param xlabel:
    :return:
    """
    xtrain = tf.reshape(xtrain, [28, 28, 1])
    # shape = xtrain.get_shape().as_list()
    # print(shape)
    xtrain = tf.image.grayscale_to_rgb( xtrain)
    shape = xtrain.get_shape().as_list()
    print(shape)
    x = tf.image.resize_images(xtrain, (224, 224))
    shape = x.get_shape().as_list()
    print(shape)
    x = tf.reshape(x, [shape[0], shape[1], shape[2]])
    y = tf.convert_to_tensor(xlabel)
    return x, y


def train():
    """
    train model
    :return:
    """
    net = Net()
    checkpoint_path = "mnist_model_3channel.h5"
    callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1,
        save_best_only=True,
        save_weights_only=True,
        period=1,
    )
    net.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    (xtrain, xtrainlabel), (xtest, xtestlabel) = Load_Mnist_Data()

    nums = xtrain.shape[0]
    batch_size = 100

    train_data = tf.data.Dataset.from_tensor_slices((xtrain, xtrainlabel))
    train_data = train_data.repeat().shuffle(nums).map(PreprocessData).batch(batch_size).prefetch(1)

    test_data = tf.data.Dataset.from_tensor_slices((xtest, xtestlabel))
    test_data = test_data.repeat().shuffle(nums).map(PreprocessData).batch(batch_size).prefetch(1)
    net.fit(train_data,
            validation_data=test_data,
            steps_per_epoch=int(nums / batch_size),
            validation_steps=5,
            callbacks=[callback],
            epochs=20)

    SaveModel2Pb(checkpoint_path)


def SaveModel2Pb(weigt_path):
    """
    save model to pb file
    :param weigt_path:
    :return:
    """
    import tensorflow.keras.backend as K
    from tensorflow.python.framework import graph_io
    from tensorflow.python.framework import graph_util

    net = Net()
    net.load_weights(weigt_path)

    orig_output_node_names = [node.op.name for node in net.outputs]
    converted_output_node_names = orig_output_node_names
    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), converted_output_node_names
    )

    output_model_name = weigt_path.replace(".h5", ".pb")
    graph_io.write_graph(constant_graph, logdir="", name=output_model_name, as_text=False)

    print("pb file path : {}".format(output_model_name))

def Load_Pb_Eval():
    """
    use pb file to test
    """
    pb_file = "mnist_model.pb"
    testfile = "test_images"

    pb_model = tf.Session()
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())  # ##load graph
        pb_model.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    for node in pb_model.graph_def.node:
        print("node: ", node.name)

    input_x = pb_model.graph.get_tensor_by_name('input_image_input_1:0')
    op = pb_model.graph.get_tensor_by_name('output_class_1/Softmax:0')

    from glob import glob
    for path in sorted(glob(testfile + "/*.png")):
        basename = os.path.basename(path)
        img = imageio.imread(path)
        img = np.expand_dims(np.array(img[...,np.newaxis]) , 0)
        ret = pb_model.run(op, {input_x: img})
        c = np.argmax(ret)
        print("base file : {} , predict label: {}".format(basename[:-3],c))

def convert_pb_to_pbtxt(filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()

        graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

        # tf.train.write_graph(graph_def, './', 'protobuf.pbtxt', as_text=True)
        tf.train.write_graph(graph_def, '', 'mnist_model.pbtxt', as_text=True)
    return



if __name__ == '__main__':
    train()
    # Load_Pb_Eval()
    # convert_pb_to_pbtxt("mnist_model.pb")




