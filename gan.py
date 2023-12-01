
import matplotlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import skimage as ski
from os import getcwd, listdir
from natsort import natsorted, ns
from random import sample, shuffle

# Last Modified: CM 11/29
class GANNet(nn.Module):

    # Layer sizes is a list of ints corresponding to the size of each layer
    def __init__(self, layer_sizes, drop_prob = 0.0):
        super(GANNet, self).__init__()

        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        self.num_layers = 0
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False)) # TODO: wtf is bias?
            nn.init.normal_(self.layers[i].weight) # Do we need this? (May be initializing weights)
            self.num_layers += 1

        self.dropout = nn.Dropout(drop_prob)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #activation function

    # Pytorch automatically handles backward for us once we have defined forward
    def forward(self, x):
        x = x.float()
        for i in range(self.num_layers - 1):
            l_0 = self.layers[i]
            x = l_0(x)
            x = self.act(x)
            x = self.dropout(x)
        l_0 = self.layers[-1]
        x = l_0(x)
        x = self.sigmoid(x)
        return x


# TODO: Write a function that tests the Generator and Discriminator classes
# Last Modified: CM 11/27
def train(G, D, training_images):
    D_learning_rate = 0.01
    G_learning_rate = 0.01
    max_epochs = 100
    loss = nn.MSELoss()

    #G= GANNet(layer_sizes= batch_size, G_learning_rate, drop_prob= 0.0) #worrying about dropout yet?
    #D = GANNet(layer_sizes= [1], D_learning_rate, drop_prob= 0.0) #learning_rate is not parameter of GANNet class

    torch_training_images = []
    for i in training_images:
        torch_training_images.append(torch.from_numpy(i))

    training_set_size = len(training_images)

    D_optimizer = optim.SGD(D.parameters(), lr=D_learning_rate)
    G_optimizer = optim.SGD(G.parameters(), lr=G_learning_rate)

    D_mean_losses = []
    G_mean_losses = []
    torch_fake_images = []

    true_labels = torch.ones(1)
    false_labels = torch.zeros(1)

    for epoch in range(max_epochs):
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()

        indices = np.arange(training_set_size)
        shuffle(indices)

        for i in range(training_set_size):

            # Generate fake image and sample true image
            noise = torch.randn(G.layer_sizes[0])
            fake_data = G(noise)

            true_data = torch_training_images[indices[i]]

            #Train Discriminator
            
            # Discriminator predictions for true and fake data
            true_data_D_out = D(true_data)
            fake_data_D_out = D(fake_data.detach())
            D_loss = loss(fake_data_D_out, false_labels) + loss(true_data_D_out, true_labels)
            D_loss.backward()
            D_optimizer.step()

            # Train Generator
            fake_data_D_out = D(fake_data)
            G_loss = loss(fake_data_D_out, true_labels)
            G_loss.backward()
            G_optimizer.step()

            print(str(epoch) + " D Fake Predict: " + str(fake_data_D_out))
            print(str(epoch) + " D Real Predict: " + str(true_data_D_out))

            if i == training_set_size - 1:
                torch_fake_images.append(fake_data.detach())
                D_mean_losses.append(torch.mean(D_loss.detach()).item())
                G_mean_losses.append(torch.mean(G_loss.detach()).item())

        export_image(torch_fake_images[epoch].numpy(), 'gen_' + str(epoch))
        export_image(torch_training_images[indices[0]].numpy(), 'true_' + str(epoch))

        print("Epoch " + str(epoch) + ": D Loss " + str(torch.mean(D_loss.detach()).item()) + ", G Loss " + str(torch.mean(G_loss.detach()).item()))


# Preconditions: 
#    - i is a float-type grayscale image vector
#    - i must be NUMPY format
#    - filename is a string, do not include .jpg in filename
# Exported images will end up in the "exported" folder
# Last Modified: CM 11/29
def export_image(i, filename):
    image = i * 255
    image = np.reshape(image, (128, 128))
    image = image.astype(np.uint8)
    ski.io.imsave(getcwd() + "/exported/" + filename + ".jpg", ski.color.gray2rgb(image))


def main():
    # CHANGE THESE VARIABLES
    # Possible choices: "dog", "cat", "corgi"
    animal_type = "corgi"
    max_training_set_size = 200

    list_files = listdir(getcwd() + "/afhq/" + animal_type)

    list_files = natsorted(list_files)
    image_list = []
    counter = 0
    for filename in list_files:
        if ".jpg" in filename:
            image = ski.io.imread(getcwd() + "/afhq/" + animal_type + "/" + filename, as_gray = True)
            # Scaling factor 0.25 –> convert 512x512 to 128x128
            image = ski.transform.rescale(image, 0.25, anti_aliasing = True)
            # Flatten matrix to vector of length 16384
            image = np.reshape(image, -1)
            image_list.append(image)
            #export_image(image, animal_type + str(counter))
            counter += 1
            if counter > max_training_set_size:
                break

    G = GANNet([2, 100, 16384])
    D = GANNet([16384, 100, 1])
    train(G, D, image_list)

if __name__ == "__main__":
    main()
