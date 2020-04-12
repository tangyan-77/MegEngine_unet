"""
unet 使用动态图进行训练
"""
import megengine as mge
import megengine.optimizer as optim
import megengine.functional as F
from megengine.data import DataLoader
from megengine.data.transform import ToMode,Compose
from megengine.data import RandomSampler
from model_unet_mge import Unet
from get_date import u_data
import time
import numpy as np

star =time.time()
path = "unet.mge"

# 读取训练数据并进行预处理
train_dataset = u_data("./data/train", order=["image", "mask"])

dataloader = DataLoader(train_dataset,
                        transform=Compose([ToMode('CHW')]),
                        sampler=RandomSampler(dataset=train_dataset, batch_size=4, drop_last=True))
# 网络和优化器的创建
le_net = Unet(1, 4)
optimizer = optim.SGD(le_net.parameters(), lr=0.05)

image = mge.tensor(dtype="float32")
label = mge.tensor(dtype="int32")
total_epochs = 100
loss_src = 1000000
for epoch in range(total_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for step, (inputs_batched, labels_batched) in enumerate(dataloader):
        labels_batched = np.squeeze(labels_batched, -1).astype(np.int32)

        image.set_value(inputs_batched)
        label.set_value(labels_batched)

        optimizer.zero_grad()  # 将参数的梯度置零
        logits = le_net(image)

        loss = F.cross_entropy_with_softmax(logits, label)

        optimizer.backward(loss)
        optimizer.step()  # 根据梯度更新参数值

        total_loss += loss.numpy().item()
        predicted = F.argmax(logits, axis=1)
        correct += ((predicted == label).sum().numpy().item()/(256*256.))
        total += label.shape[0]

    print("epoch: {:0>3}, loss {:.4f}, acc {:.4f}".format(epoch, total_loss/len(dataloader), correct/total))

    epoch_loss = total_loss/len(dataloader)
    if epoch_loss < loss_src:
        print("model saved")
        loss_src = epoch_loss
        mge.save(le_net.state_dict(), path)

print("-"*50)
print("-use time :", time.time() - star)

