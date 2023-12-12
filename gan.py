
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
    def __init__(self, layer_sizes, act, drop_prob = 0.0):
        super(GANNet, self).__init__()

        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        self.num_layers = 0
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=True))
            self.num_layers += 1

        self.dropout = nn.Dropout(drop_prob)
        self.act = act
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
def train(G, D, training_images): #change batch_size as needed; batch_size: int = 16
    D_learning_rate = 0.01
    G_learning_rate = 0.1
    max_epochs = 150
    loss = nn.BCELoss()

    torch_training_images = []
    for i in training_images:
        torch_training_images.append(torch.from_numpy(i))

    training_set_size = len(training_images)
    print("Training Set Size: " + str(training_set_size))

    D_optimizer = optim.SGD(D.parameters(), lr=D_learning_rate, momentum = 0.6)
    G_optimizer = optim.SGD(G.parameters(), lr=G_learning_rate, momentum = 0.6)

    D_mean_true_losses = []
    D_mean_fake_losses = []
    G_mean_losses = []
    torch_fake_images = []

    #may need to make these after we make the true data with batch size
    true_labels = torch.ones(1)
    false_labels = torch.zeros(1)

    for epoch in range(max_epochs):

        indices = np.arange(training_set_size)
        shuffle(indices)

        D_epoch_mean_true_loss = 0.0
        D_epoch_mean_fake_loss = 0.0
        G_epoch_mean_loss = 0.0

        for i in range(training_set_size):
            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            # Generate fake image and sample true image
            noise = torch.randn(G.layer_sizes[0]) #need to incorporate batch_size here
            #noise = torch.randint(0, 2, size=(batch_size, G.layer_sizes[0])).float() 
            fake_data = G(noise)
           
            # CHANGE with batch size- need to make new method?
            true_data = torch_training_images[indices[i]]

            # Train Generator
            fake_data_D_out = D(fake_data)
            G_loss = loss(fake_data_D_out, true_labels)
            G_loss.backward()
            G_optimizer.step()

            #Train Discriminator
            true_data_D_out = D(true_data)
            fake_data_D_out = D(fake_data.detach())
            D_true_loss = loss(true_data_D_out, true_labels)
            D_fake_loss = loss(fake_data_D_out, false_labels)

            D_true_loss.backward()
            D_fake_loss.backward()
            D_optimizer.step()
            """
            D_loss = (D_true_loss + D_fake_loss)
            D_loss.backward()
            D_optimizer.step()
            """

            D_epoch_mean_true_loss += D_true_loss.detach().item()
            D_epoch_mean_fake_loss += D_fake_loss.detach().item()
            G_epoch_mean_loss += G_loss.detach().item()

            if i == training_set_size - 1:
                torch_fake_images.append(fake_data.detach())

        export_image(torch_fake_images[-1].numpy(), 'gen_' + '{:03}'.format(epoch))
        #export_image(training_images[indices[1]], 'true_' + '{:03}'.format(epoch))

        D_mean_true_losses.append(D_epoch_mean_true_loss/training_set_size)
        D_mean_fake_losses.append(D_epoch_mean_fake_loss/training_set_size)
        G_mean_losses.append(G_epoch_mean_loss/training_set_size)

        print("Epoch " + '{:03}'.format(epoch)
                + ": DTL: " + '{:06.4f}'.format(D_mean_true_losses[-1])
                + ", DFL: " + '{:06.4f}'.format(D_mean_fake_losses[-1])
                + ", GL: " + '{:06.4f}'.format(G_mean_losses[-1]))

        #for layer in G.layers:
        #    print(layer.weight)


# Preconditions: 
#    - i is a float-type grayscale image vector
#    - i must be NUMPY format
#    - filename is a string, do not include .jpg in filename
# Exported images will end up in the "exported" folder
# Last Modified: CM 11/29
def export_image(i, filename):
    image = i * 255
    image = np.reshape(image, (image_side_length, image_side_length))
    image = image.astype(np.uint8)
    ski.io.imsave(getcwd() + "/exported/" + filename + ".jpg", ski.color.gray2rgb(image), check_contrast=False)


def main():
    # CHANGE THESE VARIABLES
    # Possible choices: "dog", "cat", "corgi"
    animal_type = "cat"
    max_training_set_size = 600
    global image_side_length
    image_side_length = 128
    gen_layers = [2, 512, 512, image_side_length*image_side_length]
    disc_layers = [image_side_length*image_side_length, 512, 64, 1]

    list_files = listdir(getcwd() + "/afhq/" + animal_type)

    list_files = natsorted(list_files)
    image_list = []
    counter = 0
    for filename in list_files:
        if ".jpg" in filename:
            image = ski.io.imread(getcwd() + "/afhq/" + animal_type + "/" + filename, as_gray = True)
            # Scale image down to image_side_length
            image = ski.transform.rescale(image, image_side_length/512.0, anti_aliasing = True)
            # Flatten matrix to vector
            image = np.reshape(image, -1)
            image_list.append(image)
            #export_image(image, animal_type + str(counter))
            counter += 1
            if counter >= max_training_set_size:
                break

    G = GANNet(gen_layers, nn.ReLU())
    D = GANNet(disc_layers, nn.LeakyReLU())
    train(G, D, image_list)

if __name__ == "__main__":
    main()
