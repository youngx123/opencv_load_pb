import numpy as np

import cv2
print("cv2 version: ", cv2.__version__)
import math
# from frame_field_learning import save_utils
from glob import  glob
# 添加 ExpLayer
class ExpLayer(object):
    def __init__(self, params, blobs):
        super(ExpLayer, self).__init__()

    def getMemoryShapes(self, inputs):
        return inputs

    def forward(self, inputs):
        return [np.exp(inputs[0])]


def get_axis_patch_count(length, stride, patch_res):
    total_double_padding = patch_res - stride
    patch_count = max(1, int(math.ceil((length - total_double_padding) / stride)))
    return patch_count

import os
if __name__ == '__main__':
    pb_path = "mnist_model.pb"
    pbtxt_path = "mnist_model.pbtxt"
    dirpath = "./test_images"
    imageNameList = glob(dirpath +"/*.png")

    net = cv2.dnn.readNetFromTensorflow(pb_path)
    layer_names = net.getLayerNames()
    for name in layer_names:
        id = net.getLayerId(name)
        layer = net.getLayer(id)
        # print("layer id : %d, type : %s, name: %s" % (id, layer.type, layer.name))
    import imageio
    sorted(imageNameList)
    for name in imageNameList:
        data = imageio.imread(name)
        data = data[..., np.newaxis]
        blob = cv2.dnn.blobFromImage(np.float32(data), 1, (224, 224), (0, 0, 0, 0))
        # blob = blob.transpose(0,2,3,1)
        net.setInput(blob)
        # print(self.PB_Net.getUnconnectedOutLayersNames())
        pred = net.forward('output_class_1/Softmax')
        c = np.argmax(pred)
        print(os.path.basename(name).split(".")[0], "->",    c    )