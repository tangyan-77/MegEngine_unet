import megengine as mge
import megengine.functional as F
from megengine.jit import trace
from model_unet_mge import Unet
import time
import cv2
import os
import numpy as np
star = time.time()

le_net = Unet(1, 4)
state_dict = mge.load("unet.mge")
le_net.load_state_dict(state_dict)

le_net.eval()  # 设置为测试模式

image = mge.tensor(dtype="float32")

path = "./data/train_l/image"
file_name = os.listdir(path)
for step, (name) in enumerate(file_name):
    image_ = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
    image_ = image_ / 255.
    image_ = image_.reshape(1, 1, 256, 256).astype(np.float32)
    image.set_value(image_)

    logits = le_net(image)
    predicted = F.argmax(logits, axis=1)
    predicted = predicted.numpy()

    predicted = predicted.transpose(1, 2, 0)
    cv2.imwrite("./result/%s" % name, predicted*80)


print("end using time:%.4f" % (time.time()-star))

