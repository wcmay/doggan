
import matplotlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import skimage as ski
from os import getcwd, listdir
from natsort import natsorted, ns
from random import sample, shuffle

class GANNet(nn.Module):

    # Layer sizes is a list of ints corresponding to the size of each layer
    def __init__(self, layer_sizes, act, final_act, drop_prob = 0.0):
        super(GANNet, self).__init__()

        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        self.num_layers = 0
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=True))
            self.num_layers += 1

        self.dropout = nn.Dropout(drop_prob)
        self.act = act
        self.final_act = final_act

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
        x = self.final_act(x)
        return x

# TODO: Write a function that tests the Generator and Discriminator classes
def train(G, D, training_images, batch_size: int = 16): #change batch_size as needed
    
    for i in training_images:
        i = (2.0*i)-1.0
    training_set_size = len(training_images)

    dataloader = DataLoader(training_images, batch_size=batch_size, shuffle=True)

    D_learning_rate = 0.0002
    G_learning_rate = 0.00005

    max_epochs = 201
    loss = nn.BCELoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    G = G.to(device)
    D = D.to(device)

    D_optimizer = optim.Adam(D.parameters(), lr=D_learning_rate, betas=(0.5, 0.99))
    G_optimizer = optim.Adam(G.parameters(), lr=G_learning_rate, betas=(0.5, 0.99))

    D_mean_true_losses = []
    D_mean_fake_losses = []
    G_mean_losses = []
    #torch_fake_images = []
    
    # true_labels = torch.ones((batch_size, 1), device=device)
    true_labels = 1-torch.abs(torch.randn((batch_size, 1), device=device)*0.01)
    false_labels = torch.zeros((batch_size, 1), device=device)

    for epoch in range(max_epochs):

        #print("\nEPOCH " + str(epoch) + "\n")

        D_epoch_mean_true_loss = 0.0
        D_epoch_mean_fake_loss = 0.0
        G_epoch_mean_loss = 0.0
        num_images_trained = 0

        for i, data in enumerate(dataloader):
            
            # Generate fake image and sample true image
            #Does not need to include last few images outside of batch_size constraints
            true_data = data.to(device)
            if true_data.size(0) != batch_size:
                break
            noise = torch.randn(batch_size, G.layer_sizes[0], device=device)
            fake_data = G(noise)

            num_images_trained += 1

            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            G.train() #sets generator to training mode

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
            # Doing the backprops individually seems to make a Significant difference
            D_true_loss.backward()
            D_fake_loss.backward()
            D_optimizer.step()

            """
            if D_loss > 0.5:
                true_data = torch_training_images[np.random.randint(training_set_size)]
                noise = torch.randn(G.layer_sizes[0])
                fake_data = G(noise).to(device)
                true_data_D_out = D(true_data)
                fake_data_D_out = D(fake_data.detach())
                D_true_loss = loss(true_data_D_out, true_labels)
                D_fake_loss = loss(fake_data_D_out, false_labels)
                D_loss = (D_true_loss + D_fake_loss)*0.5
                D_loss.backward()
                D_optimizer.step()
            """
            """
            if (i%int(training_set_size/5) == 0):
                print("Image " + str(i))
                print("\tTrue Prediction: "+str(true_data_D_out[-1]))
                print("\tTrue Loss: "+str(D_true_loss.item()))
                print("\tFake Prediction: "+str(fake_data_D_out[-1]))
                print("\tFake Loss: "+str(D_fake_loss.item()))
                print()
            """

            D_epoch_mean_true_loss += D_true_loss.detach().item()
            D_epoch_mean_fake_loss += D_fake_loss.detach().item()
            G_epoch_mean_loss += G_loss.detach().item()

        if epoch % 5 == 0:
            G.eval() #sets generator to evaluation mode
            export_image(sample(training_images, 1)[0], 'true_' + '{:03}'.format(epoch))
            with torch.no_grad():
                for i in range(5):
                    noise = torch.randn(G.layer_sizes[0])
                    fake_data = G(noise).detach()
                    export_image(0.5*((fake_data.numpy())+1.0), 'gen_' + '{:03}'.format(epoch) + '_' + '{:02}'.format(i))


        D_mean_true_losses.append(D_epoch_mean_true_loss/num_images_trained)
        D_mean_fake_losses.append(D_epoch_mean_fake_loss/num_images_trained)
        G_mean_losses.append(G_epoch_mean_loss/num_images_trained)

        print("Epoch " + '{:03}'.format(epoch)
                + ": DTL: " + '{:06.4f}'.format(D_mean_true_losses[-1])
                + ", DFL: " + '{:06.4f}'.format(D_mean_fake_losses[-1])
                + ", GL: " + '{:06.4f}'.format(G_mean_losses[-1]))


# Preconditions: 
#    - i is a float-type grayscale image vector
#    - i must be NUMPY format
#    - filename is a string, do not include .jpg in filename
# Exported images will end up in the "exported" folder
def export_image(i, filename):
    image = i * 255
    image = np.reshape(image, (image_side_length, image_side_length))
    image = image.astype(np.uint8)
    ski.io.imsave(getcwd() + "/exported/" + filename + ".jpg", ski.color.gray2rgb(image), check_contrast=False)


def main():
    # CHANGE THESE VARIABLES
    # Possible choices: "dog", "cat", "corgi"
    animal_type = "corgi"
    max_training_set_size = 600
    global image_side_length
    image_side_length = 128
    gen_layers = [32, 512, 512, image_side_length*image_side_length]
    disc_layers = [image_side_length*image_side_length, 512, 256, 64, 1]

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
            counter += 1
            if counter >= max_training_set_size:
                break

    G = GANNet(gen_layers, nn.LeakyReLU(), nn.Tanh(), drop_prob=0.0)
    D = GANNet(disc_layers, nn.LeakyReLU(), nn.Sigmoid(), drop_prob=0.0)
    train(G, D, image_list)

if __name__ == "__main__":
    main() 
