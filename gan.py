
import matplotlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import skimage as ski
import os
from natsort import natsorted, ns

# Last Modified: CM 11/29
class GANNet(nn.Module):

    # Layer sizes is a list of ints corresponding to the size of each layer
    # Eg: [6, 4, 1]
    def __init__(self, layer_sizes, drop_prob = 0.0):
        super(GANNet, self).__init__()

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        self.dropout = nn.Dropout(drop_prob)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #activation function

    # Pytorch automatically handles backward for us once we have defined forward
    def forward(self, x):
        for i in range(self.num_layers - 2):
            l_0 = layers[i]
            x = l_0(x)
            x = self.act(x)
            x = self.dropout(x)
        l_0 = layers[-1]
        x = l_0(x)
        x = self.sigmoid(x)
        return x


# TODO: Write a function that tests the Generator and Discriminator classes
# Last Modified: CM 11/27
def train(G, D):
    d_learning_rate = 0.01
    g_learning_rate = 0.01
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate)
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate)


# Preconditions: 
#    - i is a float-type grayscale image vector (must be in non-flattened matrix form for right now?)
#    - filename is a string, DO NOT INCLUDE .jpg in filename
# Exported images will end up in the "exported" folder
# Last Modified: CM 11/29
def export_image(i, filename):
    image = i * 255
    image = image.astype(np.uint8)
    ski.io.imsave(os.getcwd() + "/exported/" + filename + ".jpg", ski.color.gray2rgb(image))


def main():
    # CHANGE THESES VARIABLES
    # Possible choices: "dog", "cat", "corgi"
    animal_type = "corgi"
    max_training_set_size = 60

    list_files = os.listdir(os.getcwd() + "/afhq/" + animal_type)

    list_files = natsorted(list_files)
    image_list = []
    counter = 0
    for filename in list_files:
        if ".jpg" in filename:
            image = ski.io.imread(os.getcwd() + "/afhq/" + animal_type + "/" + filename, as_gray = True)
            # Scaling factor 0.25 –> convert 512x512 to 128x128
            image = ski.transform.rescale(image, 0.25, anti_aliasing = True)
            image_list.append(image)
            export_image(image, animal_type + str(counter))
            counter += 1
            if counter > max_training_set_size:
                break

    G = GANNet([10, 128])
    D = GANNet([128, 1])
    #train(G, D)

if __name__ == "__main__":
    main()
