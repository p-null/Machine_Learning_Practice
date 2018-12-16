import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(1)
#x = np.linspace(-1, 1, 200)[:, None]       # [batch, 1]
#y = x ** 2                                  # [batch, 1]
learning_rate = 0.001

data = pd.read_csv('data.csv', header=None)
train_X = data.iloc[:, :2].values
batch_size = train_X.shape[0]
#x = train_X.reshape(batch_size, 2, 1)
train_y = data.iloc[:, 2].values.reshape(batch_size, 1)


def tanh(x):
    return np.tanh(x)


def derivative_tanh(x):
    return 1 - tanh(x)**2



def iter_data(*datas, n_batch=8, truncate=False, verbose=False, max_batches=float("inf")):
    n = len(datas[0])
    if truncate:
        n = (n // n_batch) * n_batch
    n = min(n, max_batches * n_batch)
    n_batches = 0
    if verbose:
        f = sys.stderr
    else:
        f = open(os.devnull, 'w')
    for i in tqdm(range(0, n, n_batch), total=n // n_batch, file=f, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i + n_batch]
        else:
            yield (d[i:i + n_batch] for d in datas)
        n_batches += 1



hidden_nodes = 56
w1 = np.random.uniform(0, 1, (2, hidden_nodes))
w2 = np.random.uniform(0, 1, (hidden_nodes, hidden_nodes))
w3 = np.random.uniform(0, 1, (hidden_nodes, 1))

b1 = np.full((1, hidden_nodes), 0.01)
b2 = np.full((1, hidden_nodes), 0.01)
b3 = np.full((1, 1), 0.01)


decay_rate_1 = 0.9
decay_rate_2 = 0.99
lr = learning_rate
l2_m = 0
l1_m = 0
l2_v = 0
l1_v = 0

first_m = [0,0,0]
second_m = [0,0,0]


t = 0
epsilon = 10e-08

for x, y in iter_data(train_X, train_y):
    a1 = x
    z2 = a1.dot(w1) + b1
    a2 = tanh(z2)
    z3 = a2.dot(w2) + b2
    a3 = tanh(z3)
    z4 = a3.dot(w3) + b3

    cost = np.sum((z4 - y)**2)/2

    # backpropagation
    z4_delta = z4 - y
    dw3 = a3.T.dot(z4_delta)
    db3 = np.sum(z4_delta, axis=0, keepdims=True)

    z3_delta = z4_delta.dot(w3.T) * derivative_tanh(z3)
    dw2 = a2.T.dot(z3_delta)
    db2 = np.sum(z3_delta, axis=0, keepdims=True)

    z2_delta = z3_delta.dot(w2.T) * derivative_tanh(z2)
    dw1 = x.T.dot(z2_delta)
    db1 = np.sum(z2_delta, axis=0, keepdims=True)

    # update parameters
    for param, gradient in zip([w1, w2, w3, b1, b2, b3], [dw1, dw2, dw3, db1, db2, db3]):
        layer = 1

        t += 1  # Increment Time Step

        # Computing 1st and 2nd moment for each layer
        first_m[layer] = first_m[layer] * decay_rate_1 + (1 - decay_rate_1) * gradient

        second_m[layer] = second_m[layer] * decay_rate_2 + (1 - decay_rate_2) * (gradient ** 2)

        # Computing bias-corrected moment
        l2_m_corrected = first_m[layer] / (1 - (decay_rate_1 ** t))
        l2_v_corrected = second_m[layer] / (1 - (decay_rate_2 ** t))

        # Update Weights
        param_update = l2_m_corrected / (np.sqrt(l2_v_corrected) + epsilon)

        param -= learning_rate * param_update
        layer += 1
    print(cost)

plt.plot(x, z4)
plt.show()