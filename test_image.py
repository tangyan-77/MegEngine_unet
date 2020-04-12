import megengine as mge
import megengine.functional as F
from megengine.jit import trace
from model_unet_mge import Unet
import time
import cv2
import os
import numpy as np
star = time.time()

@trace
def eval_func(data, *, net):
    logits = net(data)
    return logits


le_net = Unet(1, 4)
state_dict = mge.load("unet.mge")
le_net.load_state_dict(state_dict)

trace.enabled = True  # 开启trace，使用静态图模式
le_net.eval()  # 设置为测试模式

data = mge.tensor()

path = "./data/train/image"
file_name = os.listdir(path)
for step, (name) in enumerate(file_name):

    image = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
    image = image / 255.
    image = image.reshape(1, 1, 256, 256).astype(np.float32)

    print(image.shape, name)

    # data.set_value(image)

    logits = eval_func(image, net=le_net)
    predicted = F.argmax(logits, axis=1)
    predicted = predicted.numpy()

    predicted = predicted.transpose(1, 2, 0)
    cv2.imwrite("./result/%s" % name, predicted*80)


print("end using time:%.4f" % (time.time()-star))

