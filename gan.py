
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Last Modified: CM 11/27
class Generator(nn.Module):

	# Layer sizes is a list of ints corresponding to the size of each layer
	# Eg: [6, 4, 1]
    def __init__(self, input_size, layer_sizes, output_size, drop_prob = 0.5):
        super(Generator, self).__init__()

        self.num_layers = len(layer_sizes)
        self.layers = []
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        for i in range(self.num_layers - 1):
        	self.layers.append(nn.Linear(layer_sizes[i+1], layer_sizes[i+2]))
        self.layers.append(nn.Linear(layer_sizes[-1], output_size))

        self.dropout = nn.Dropout(drop_prob)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #activation function

    # Pytorch automatically handles backward for us once we have defined forward
    def forward(self, x):
        for i in range(self.num_layers - 1):
        	x = layers[i](x)
        	x = self.act(x)
        	x = self.dropout(x)
        x = layers[-1](x)
        x = self.sigmoid(x)
        return x

# Last Modified: CM 11/27
class Discriminator(nn.Module):

	# Layer sizes is a list of ints corresponding to the size of each layer
	# Eg: [6, 4, 1]
	# Discriminator output size should always be 1 (bc it's just true or false)
    def __init__(self, input_size, layer_sizes, drop_prob = 0.5):
        super(Discriminator, self).__init__()

        self.num_layers = len(layer_sizes)
        self.layers = []
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        for i in range(self.num_layers - 1):
        	self.layers.append(nn.Linear(layer_sizes[i+1], layer_sizes[i+2]))
        self.layers.append(nn.Linear(layer_sizes[-1], 1))

        self.dropout = nn.Dropout(drop_prob)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # Pytorch automatically handles backward for us once we have defined forward
    def forward(self, x):
        for i in range(self.num_layers - 1):
        	x = layers[i](x)
        	x = self.act(x)
        	x = self.dropout(x)
        x = layers[-1](x)
        x = self.sigmoid(x)
        return x

# TODO: Write a function that tests the Generator and Discriminator classes
# Last Modified: CM 11/27
def train(G, D):
    d_learning_rate = 0.01
    g_learning_rate = 0.01
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate)
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate)

def main():
	G = Generator(1, [1], 1)
	D = Discriminator(1, [1])
    train(G, D)

if __name__ == "__main__":
    main()


