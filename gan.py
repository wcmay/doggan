
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

# Last Modified: CM 11/29
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
# Last Modified: CM 11/27
def train(G, D, training_images, batch_size: int = 16): #change batch_size as needed
    
    dataloader = DataLoader(training_images, batch_size=batch_size, shuffle=True)

    training_set_size = len(training_images)
   
    D_learning_rate = 0.001
    G_learning_rate = 0.0001
    max_epochs = 201
    loss = nn.BCELoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    G = G.to(device)
    D = D.to(device)

    # t = []
    # # Normalizing values
    # for i in training_images:
    #     t.append((2.0 * torch.from_numpy(i).to(device))-1.0)
    # training_set_size = len(t)
    # torch_training_images = torch.Tensor(t,dtype=torch.int)

    # print("Training Set Size: " + str(training_set_size))
    # torch_training_images = torch.Tensor(training_set_size, D.layer_sizes[0])
    # print(torch_training_images.size())
    # print(len(t))
    # torch_training_images = torch.cat(t).float()

    D_optimizer = optim.Adam(D.parameters(), lr=D_learning_rate, betas=(0.5,0.99))
    G_optimizer = optim.Adam(G.parameters(), lr=G_learning_rate, betas=(0.5,0.99))

    D_mean_true_losses = []
    D_mean_fake_losses = []
    G_mean_losses = []
    #torch_fake_images = []

    #may need to make these after we make the true data with batch size
    #true_labels = torch.ones(batch_size)
    
    # true_labels = torch.ones((batch_size, 1), device=device)
    true_labels = 1-torch.abs(torch.randn((batch_size, 1), device=device)*0.01)
    false_labels = torch.zeros((batch_size, 1), device=device)

    for epoch in range(max_epochs):

        #print("\nEPOCH " + str(epoch) + "\n")

        # indices = np.arange(training_set_size) #changed
        # #shuffle(indices)

        D_epoch_mean_true_loss = 0.0
        D_epoch_mean_fake_loss = 0.0
        G_epoch_mean_loss = 0.0

        #for i in range(int(training_set_size/batch_size)):
        for i, data in enumerate(dataloader):
            
            true_data = data.to(device)
            #Does not need to include last few images outside of batch_size constraints
            if true_data.size(0) != batch_size:
                break
            #print(true_data[1].size())

            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            G.train() #sets generator to training mode
            # true_labels = torch.reshape(1-torch.abs(torch.randn(batch_size)*0.01), (16,1)) #very slightly noisy true labels

            # Generate fake image and sample true image
            #noise = torch.randn(G.layer_sizes[0]) #need to incorporate batch_size here
            #noise = torch.randn(batch_size)
            noise = torch.randn(batch_size, G.layer_sizes[0], device=device)
            #noise = torch.randint(0, 2, size=(batch_size, G.layer_sizes[0])).float() 
            fake_data = G(noise)
           
            # true_data = torch_training_images[indices[i*batch_size]:indices[(i+1)*batch_size]]
            # #torch.FloatTensor(
            # print(true_data.size())

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

            D_loss = (D_true_loss + D_fake_loss)*0.5
            D_loss.backward()
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
                print("\tTrue Prediction: "+str(true_data_D_out.item()))
                print("\tTrue Loss: "+str(D_true_loss.item()))
                print("\tFake Prediction: "+str(fake_data_D_out.item()))
                print("\tFake Loss: "+str(D_fake_loss.item()))
                print()
            """

            D_epoch_mean_true_loss += D_true_loss.detach().item()
            D_epoch_mean_fake_loss += D_fake_loss.detach().item()
            G_epoch_mean_loss += G_loss.detach().item()

            #if i == training_set_size - 1:
            #    torch_fake_images.append(fake_data.detach())

        if epoch % 5 == 0:
            G.eval() #sets generator to evaluation mode
            with torch.no_grad():
                for i in range(5):
                    noise = torch.randn(G.layer_sizes[0])
                    fake_data = G(noise).detach()
                    export_image(0.5*((fake_data.numpy())+1.0), 'gen_' + '{:03}'.format(epoch) + '_' + '{:02}'.format(i))


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
    animal_type = "corgi"
    max_training_set_size = 600
    global image_side_length
    image_side_length = 128
    gen_layers = [32, 256, 512, image_side_length*image_side_length]
    disc_layers = [image_side_length*image_side_length, 512, 128, 64, 1]

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

    G = GANNet(gen_layers, nn.LeakyReLU(), nn.Tanh())
    D = GANNet(disc_layers, nn.LeakyReLU(), nn.Sigmoid())
    train(G, D, image_list)

if __name__ == "__main__":
    main()
