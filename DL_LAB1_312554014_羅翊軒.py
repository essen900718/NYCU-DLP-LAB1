import numpy as np
import matplotlib.pyplot as plt

# generate linear dataset
def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

#generate XOR dataset
def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

# visualization for result
def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

class Layer:
    def __init__(self, n_currentlayer, n_nextlayer) -> None:
        
        def weight_init(n_lastlayer, n_currentlayer):
            return np.random.randn(n_lastlayer, n_currentlayer)
        
        self.weights = weight_init(n_currentlayer, n_nextlayer)
        self.values = np.zeros((n_nextlayer,), dtype=np.float64)
        self.outputs = np.zeros((n_nextlayer,), dtype=np.float64)
        self.gradients = np.array([], dtype=np.float64)
    
class ANN:
    def __init__(self, n_layers, learningRate=0.1, activation="sigmoid") -> None:
        def layer_init(n_layers):
            layers = []
            for i in range(0, len(n_layers)-1):
                layers.append(Layer(n_layers[i], n_layers[i+1]))
            return layers
        self.inputs = []
        self.n_layers = n_layers
        self.layers = layer_init(n_layers)
        self.layerSize = len(self.layers)
        self.learningRate = learningRate
        self._lossFunction = self.mse_loss

        if activation == "sigmoid":
            self._activation = self.sigmoid
            self.derivative_activation = self.derivative_sigmoid
        elif activation == "without":
            self._activation = self.without
            self.derivative_activation = self.derivative_without
        else:
            print("Error !!!")
            exit()
    
    def forwardPass(self, x):
        
        for i in range(self.layerSize):
            if i == 0:
                self.layers[i].values = np.dot(x, self.layers[i].weights)
            else:
                self.layers[i].values = np.dot(self.layers[i-1].outputs, self.layers[i].weights)
            self.layers[i].outputs = self._activation(self.layers[i].values)

        return self.layers[i].outputs
    
    def backwardPass(self, error):
     
        delta = np.array(error * self.derivative_activation(self.layers[self.layerSize-1].outputs))
        
        for i in range(self.layerSize-1, -1, -1):
            if i == self.layerSize-1:
                self.layers[i].gradients = delta
            else:
                delta = np.dot(self.layers[i+1].gradients, self.layers[i+1].weights.transpose())
                delta = delta * self.derivative_activation(self.layers[i].outputs)
                self.layers[i].gradients = delta

    def update_weights(self):
      
        for i in range(self.layerSize):
            if i == 0: 
                self.layers[i].weights += -self.learningRate * np.dot(self.inputs.transpose(), self.layers[i].gradients)
            else:
                self.layers[i].weights += -self.learningRate * np.dot(self.layers[i-1].outputs.transpose(), self.layers[i].gradients)

    def train(self, x, y, epoch):

        train_meta = []
        for i in range(epoch):
            acc_epoch = 0.0
            loss_epoch = 0.0
            num = len(x)
            for j, (xi, yi) in enumerate(zip(x, y)):
                self.inputs = np.array([xi])
                yhat = self.forwardPass([xi])
                error = yhat - [yi]
                self.backwardPass(error)
                self.update_weights()
                loss = self._lossFunction(xi, yi)
                pred_y = 1.0 if yhat > 0.5 else 0.0
                acc = self.calculate_acc(yi, pred_y)
                print("[ Epoch{} / Step{} ] pred_y:{}, actual_y:{}, loss:{:.6f}, acc:{}".format(i+1, j+1, yhat.flatten(), [yi], loss, acc))
                acc_epoch += acc
                loss_epoch += loss
            acc_epoch /= num
            loss_epoch /= num
            train_meta.append([i+1, loss_epoch, acc_epoch])
        return train_meta
    
    def test(self, x, y):
        pred_y = model.predict(x)
        print(pred_y)
        pred_y_ = np.atleast_2d([1 if yi > 0.5 else 0 for yi in pred_y]).transpose()
        acc = self.calculate_acc(y, pred_y_)
        print("Accuracy: {}%".format(acc*100))
        return pred_y, pred_y_, acc
    
    def predict(self, x):
        return self.forwardPass(x)
    
    # MSE loss
    def mse_loss(self,x, y):
        yhat = self.predict(x)
        return (np.square(yhat - y)).mean()
    
    # sigmoid activation function
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
    def derivative_sigmoid(self, x):
        return np.multiply(x, 1.0 - x)
    
    # without activation function
    def without(self, x):
        return x
    def derivative_without(self, x):
        return 0
    
    def calculate_acc(self, y, yhat):
        num = len(y)
        return (y == yhat).sum() / num

if __name__ == '__main__':

    # generate dataset
    x, y = generate_linear(n=100)
    #x, y = generate_XOR_easy()
    
    # visualization for dataset
    # plt.scatter(x[:,0], x[:,1], c = ["blue" if c else "red" for c in y])
    # plt.show()

    model = ANN([2, 3, 3, 1], 0.1, "sigmoid")

    train_meta = np.array(model.train(x, y, 2000))
    pred_y, pred_y_, acc = model.test(x, y)
    
    # visualization for train info
    plt.figure("Loss")
    plt.plot(train_meta[:,0], train_meta[:,1], label='loss')
    plt.show()

    plt.figure("Accuracy")
    plt.plot(train_meta[:,0], train_meta[:,2], label='acc')
    plt.show()

    #visualization for test result
    show_result(x, y, pred_y_)


    





