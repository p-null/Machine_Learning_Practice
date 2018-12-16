import numpy as np
import pandas as pd
from utils.math import sigmoid, tanh, gelu, swish
np.seterr(over='ignore')

Act_Func = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'gelu': gelu,
    'swish': swish
}

class NeuralNetwork():
    def __init__(self, activation:str = 'sigmoid'):
        self.act_func = Act_Func[activation]
        np.random.seed(42)  # deterministic 
        self.weights = {}  # weights between layers
        self.biases = {}
        self.num_layers = 1  # Set initial number of layer to one (input layer)
        self.adjustments = {}  # Adjustements of weights

    def add_layer(self, shape):
        # shape of weight + bais
        # weights[1] means weights between layer 1 and layer 2
        self.weights[self.num_layers] = np.random.random(shape)
        self.biases[self.num_layers] = np.random.random((1, shape[1]))
        # Adjustments only apply to the weight (w) part
        self.adjustments[self.num_layers] = np.zeros(shape)
        self.num_layers += 1

    def predict(self, data):

        return __forward_propagate(data)[-1]

    def __forward_propagate(self, data):
        # `output` hold activation values of each layer
        output = {}
        output[1] = data
        for layer in range(1, self.num_layers):
            net = np.dot(output[layer], self.weights[layer]) + self.biases[layer] # + self.biases[layer]
            output[layer+1] = self.act_func(net)
        return output

    def sum_squared_error(self, outputs, targets):
        return 0.5 * np.mean(np.sum(np.power(outputs - targets, 2), axis=1))

    def __back_propagate(self, output, target):
        deltas = {}
        # Delta of output Layer
        deltas[self.num_layers] = output[self.num_layers] - target

        # Delta of hidden Layers
        for layer in reversed(range(2, self.num_layers)):  # All layers except input/output
            a_val = output[layer].T
            weights = self.weights[layer]

            prev_deltas = deltas[layer+1]
            deltas[layer] = np.multiply(np.dot(weights, prev_deltas), self.act_func(a_val,True))

        for layer in range(1, self.num_layers):
            self.adjustments[layer] += np.dot(deltas[layer+1], output[layer]).T



    def __gradient_descent(self, batch_size, learning_rate):
        # updating weight and bias
        for layer in range(1, self.num_layers):
            partial_d = (1/batch_size) * self.adjustments[layer]
            self.weights[layer] += learning_rate * -partial_d
            self.biases[layer] += learning_rate*1e-3 * -partial_d[-1, :]


    def train(self, inputs, targets, num_epochs, learning_rate=1, stop_accuracy=1e-5):
        error = []
        for iteration in range(num_epochs):

            for i in range(len(inputs)):
                x = inputs[i]
                y = targets[i]
                output = self.__forward_propagate(x)
                loss = self.sum_squared_error(output[self.num_layers], y)
                error.append(loss)
                # calculate adjustements
                self.__back_propagate(output, y)
                
            # apply the adjustments
            self.__gradient_descent(i, learning_rate)
            #error.append(np.mean(iter_error))
            if np.mean(error[-(i+1):]) < stop_accuracy and iteration > 0:
                break

        return(np.asarray(error), iteration+1)



if __name__ == "__main__":

    nn = NeuralNetwork(activation='tanh')

    nn.add_layer((2, 256))
    nn.add_layer((256, 1))

    data = pd.read_csv('data.csv',header=None)
    train_X =  data.iloc[:,:2].values
    batch_size = train_X.shape[0]
    #train_X = train_X.reshape(batch_size,2,1)
    train_y = data.iloc[:,2].values.reshape(batch_size,1)

    error, iteration = nn.train(train_X, train_y, 50)

    print('Error = ', np.mean(error[-(batch_size+1):]))
    print('Epoches = ', iteration)

    # nn.predict(test_data)
