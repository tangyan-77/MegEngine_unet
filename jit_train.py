import megengine as mge
import megengine.optimizer as optim
import megengine.functional as F
from megengine.data import DataLoader, RandomSampler
from megengine.data.transform import ToMode,Compose
from megengine.jit import trace
from model_unet_mge import Unet
from get_date import u_data
import time
import numpy as np

star = time.time()
path = "unet_j.mge"

@trace
def train_func(data, label, net=None, optimizer = None):
    net.train()
    pred = net(data)
    loss = F.cross_entropy_with_softmax(pred, label)
    optimizer.backward(loss)
    return pred, loss


train_dataset = u_data("./data/train", order=["image", "mask"])
dataloader = DataLoader(train_dataset,
                        transform=Compose([ToMode('CHW')]),
                        sampler=RandomSampler(dataset=train_dataset, batch_size=4, drop_last=True))

unet = Unet(1, 4)
optimizer = optim.SGD(unet.parameters(), lr=0.05)

trace.enabled = True

total_epochs = 50
loss_src = 100000000
for epoch in range(total_epochs):
    total_loss = 0
    correct = 0
    total = 0
    sta = time.time()

    for step, (inputs_batched, labels_batched) in enumerate(dataloader):

        optimizer.zero_grad()
        labels_batched = np.squeeze(labels_batched, -1).astype(np.int32)

        logits, loss = train_func(inputs_batched, labels_batched, optimizer=optimizer, net=unet)

        optimizer.step()

        total_loss += loss.numpy().item()
        predicted = np.argmax(logits, axis=1)
        correct += ((predicted == labels_batched).sum().item()/(256*256.))
        total += labels_batched.shape[0]

    print("==>>epoch: {:0>3}, loss: {:.4f}, acc: {:.4f},  time: {:.4f}".format(
        epoch, total_loss/len(dataloader), correct/total, (time.time()-sta)))
    epoch_loss = total_loss/len(dataloader)
    if epoch_loss < loss_src:
        print("model saved")
        loss_src = epoch_loss
        mge.save(unet.state_dict(), path)

print("-"*50)
print("-use time :", time.time() - star)


