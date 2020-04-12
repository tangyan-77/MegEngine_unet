import megengine as mge
import megengine.functional as F
from megengine.data import DataLoader
from megengine.data.transform import ToMode, Compose
from megengine.data import RandomSampler
from megengine.jit import trace
from model_unet_mge import Unet
from get_date import u_data
import time
import cv2
import numpy as np
star = time.time()



@trace
def eval_func(data, *, net):  # *号前为位置参数，*号后为关键字参数
    logits = net(data)
    return logits


train_dataset = u_data("./data/train", order=["image"])

dataloader = DataLoader(
    train_dataset,
    transform=Compose([ToMode('CHW')]),
    sampler=RandomSampler(dataset=train_dataset, batch_size=4, drop_last=True))
# 网络和优化器的创建
le_net = Unet(1, 4)
state_dict = mge.load("unet_j.mge")  # 将参数加载到网络
le_net.load_state_dict(state_dict)

trace.enabled = True  # 开启trace，使用静态图模式
le_net.eval()  # 设置为测试模式


data = mge.tensor()
label = mge.tensor(dtype="int32")
correct = 0
total = 0
for step, (inputs_batched) in enumerate(dataloader):
    data.set_value(inputs_batched)

    logits = eval_func(data, net=le_net)
    predicted = F.argmax(logits, axis=1)
    predicted = predicted.numpy()

    predicted = predicted.transpose(1, 2, 0)
    for n in range(predicted.shape[2]):
        name = step*predicted.shape[2]+n
        print(name)
        cv2.imwrite("./result/%d.png"%(name), predicted[:, :, n]*80)

print("end using time:%.4f"%(time.time()-star))

