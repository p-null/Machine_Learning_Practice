import numpy as np
import pandas as pd
import neuralnets as nn
import matplotlib.pyplot as plt#, draw, show
from tqdm import tqdm
np.random.seed(42)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        w_init = nn.init.RandomUniform(low=0, high=0.001)
        b_init = nn.init.Constant(0.001)

        self.l1 = nn.layers.Dense(2, 256, nn.act.sigmoid, w_init, b_init)
        self.l2 = nn.layers.Dense(256, 256, nn.act.sigmoid, w_init, b_init)
        self.out = nn.layers.Dense(256, 1, w_initializer=w_init, b_initializer=b_init)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        o = self.out(x)
        return o


net = Net()
opt = nn.optim.Adam(net.params, lr=0.01)
loss_fn = nn.losses.MSE()


def iter_data(*datas, n_batch=50, truncate=False,  max_batches=float("inf")):
    n = len(datas[0])
    if truncate:
        n = (n // n_batch) * n_batch
    n = min(n, max_batches * n_batch)
    n_batches = 0

    for i in tqdm(range(0, n, n_batch), total=n // n_batch, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i + n_batch]
        else:
            yield (d[i:i + n_batch] for d in datas)
        n_batches += 1


fig, ax = plt.subplots()

data = pd.read_csv('data.csv', header=None)
train_X = data.iloc[:, :2].values
num_example = train_X.shape[0]
train_y = data.iloc[:, 2].values.reshape(num_example, 1)

for epoch in range(550):
    perm_data = np.random.permutation(data.values)
    perm_X = perm_data[:,:-1]
    perm_y = perm_data[:,-1].reshape((perm_X.shape[0],1))
    for x, y in iter_data(perm_X, perm_y):
        o = net.forward(x)
        loss = loss_fn(o, y)
        net.backward(loss)
        opt.step()
    o = net.forward(perm_X)
    loss = loss_fn(o, perm_y)

    print("epoch: %i | loss: %.5f" % (epoch, loss.data))

o = net.forward(train_X)
plt.scatter(train_X[:,1], train_y, s=20)
plt.scatter(train_X[:,1], o.data, s=5)
#plt.plot(train_X[:,1], o.data, c="red", lw=3)
plt.show()