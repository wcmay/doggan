
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
            x = self.dropout(x)
            l_0 = self.layers[i]
            x = l_0(x)
            x = self.act(x)
        l_0 = self.layers[-1]
        x = l_0(x)
        x = self.final_act(x)
        return x

# TODO: Write a function that tests the Generator and Discriminator classes
<<<<<<< Updated upstream
def train(G, D, training_images, avg_pxl_arr, avg_pxl_float, image_side_length, batch_size: int = 16): #change batch_size as needed
=======
def train(G, D, training_images, avg_pxl, image_side_length, batch_size): #change batch_size as needed
>>>>>>> Stashed changes
    
    training_set_size = len(training_images)
    print("Training Set Size: " + str(training_set_size))

    dataloader = DataLoader(training_images, batch_size=batch_size, shuffle=True)

    D_learning_rate = 0.00006
    G_learning_rate = 0.00008

    max_epochs = 201
    loss = nn.BCELoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    G = G.to(device)
    D = D.to(device)

    D_optimizer = optim.Adam(D.parameters(), lr=D_learning_rate, betas=(0.0, 0.99))
    G_optimizer = optim.Adam(G.parameters(), lr=G_learning_rate, betas=(0.5, 0.99))

    D_mean_true_losses = []
    D_mean_fake_losses = []
    G_mean_losses = []
    #torch_fake_images = []
    
    false_labels = torch.zeros((batch_size, 1), device=device)

    for epoch in range(max_epochs):

        #print("\nEPOCH " + str(epoch) + "\n")

        D_epoch_mean_true_loss = 0.0
        D_epoch_mean_fake_loss = 0.0
        G_epoch_mean_loss = 0.0
        num_images_trained = 0

        image_mse_total = 0
        fake_data_dev = []
        fake_to_real_dev = 0

        for i, data in enumerate(dataloader):

            true_labels = 1-torch.abs(torch.randn((batch_size, 1), device=device)*0.07)
            
            # Generate fake image and sample true image
            # Does not need to include last few images outside of batch_size constraints
            true_data = (data.to(device)*2.0)-1.0
            if true_data.size(0) != batch_size:
                break
            noise = torch.randn(batch_size, G.layer_sizes[0], device=device)
            fake_data = G(noise)

            num_images_trained += 1

            # Keep track of standard deviation within fake data
            fake_list = fake_data.detach().cpu().numpy()
            fake_data_dev.append(np.std(fake_list))

            #For MSE Loss and STD
            for i in range(batch_size):
                f = np.mean(fake_data[i].detach().cpu().numpy())
                # Calculate MSE for each image, ultimately averaging for the epoch
                image_mse_total += image_mse(avg_pxl_arr, f, image_side_length)
                # Calculate pixelation difference for each fake image compared to the training images
                # Use this to get standard deviation at end of epoch
                fake_to_real_dev += np.subtract(f, avg_pxl_float)**2
                # print(f)
                # print(len(f))

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

            D_epoch_mean_true_loss += D_true_loss.detach().item()
            D_epoch_mean_fake_loss += D_fake_loss.detach().item()
            G_epoch_mean_loss += G_loss.detach().item()

        if epoch % 5 == 0:
            G.eval() #sets generator to evaluation mode
            with torch.no_grad():
                for i in range(5):
                    noise = torch.randn(G.layer_sizes[0])
                    fake_data = G(noise).detach()
                    export_image(0.5*(fake_data.numpy()+1.0), 'gen_' + '{:03}'.format(epoch) + '_' + '{:02}'.format(i))

        D_mean_true_losses.append(D_epoch_mean_true_loss/num_images_trained)
        D_mean_fake_losses.append(D_epoch_mean_fake_loss/num_images_trained)
        G_mean_losses.append(G_epoch_mean_loss/num_images_trained)
        
        # Calculates standard deviation of fake data
        fake_stan_dev = np.mean(fake_data_dev)
        
        image_iterations = training_set_size // batch_size
        image_iterations *= batch_size
        # Average mean squared error between all images in this epoch
        image_mse_mean = image_mse_total / image_iterations 

        # Standard deviation of fake images compared to that of training images
        fake_to_real_dev /= image_iterations
        fake_to_real_dev = np.sqrt(fake_to_real_dev)
    

        print("Epoch " + '{:03}'.format(epoch)
                + ": DTL: " + '{:06.4f}'.format(D_mean_true_losses[-1])
                + ", DFL: " + '{:06.4f}'.format(D_mean_fake_losses[-1])
                + ", GL: " + '{:06.4f}'.format(G_mean_losses[-1])
                + ", PVD: " + '{:06.4f}'.format(image_mse_mean) #PVD = pixel value difference
                + ", FSTD: " + '{:06.4f}'.format(fake_stan_dev)
                + ", FRSTD: " + '{:06.4f}'.format(fake_to_real_dev)) 


def evaluate_finished_model(G):
    G.eval()
    noise = torch.randn(50, G.layer_sizes[0], device=cpu)
    fake_data = G(noise).detach()
    counter = 0
    for i in fake_data:
        export_image(0.5*(i.numpy()+1.0), 'finished_' + str(counter))
        counter += 1








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

# Comparing pixel intensities of images
def image_mse(avg_pxl, fake_batch_pics, image_side_length):
	diff = np.sum((avg_pxl.astype("float") - fake_batch_pics.astype("float")) ** 2)
	diff /= float(image_side_length ** 2)
	# lower error is more "similar" in pixel intensity
	return diff

def main():
    # CHANGE THESE VARIABLES
    # Possible choices: "dog", "cat", "corgi"
    animal_type = "cat"
    max_training_set_size = 99999
    global image_side_length
    image_side_length = 128
    gen_layers = [32, 128, 512, 512, 512, image_side_length*image_side_length]
    disc_layers = [image_side_length*image_side_length, 512, 512, 256, 128, 64, 1]

    list_files = listdir(getcwd() + "/afhq/" + animal_type)

    list_files = natsorted(list_files)
    image_list = []
    avg_pxl_arr = np.zeros(image_side_length**2)
    counter = 0

    for filename in list_files:
        if ".jpg" in filename:
            image = ski.io.imread(getcwd() + "/afhq/" + animal_type + "/" + filename, as_gray = True)
            # Scale image down to image_side_length
            image = ski.transform.rescale(image, image_side_length/512.0, anti_aliasing = True)
            # Flatten matrix to vector
            image = np.reshape(image, -1)
            image_list.append(image)
            
            # Keeps track of pixelation array of all pixels and average overal pixelation value
            avg_pxl_arr = np.add(avg_pxl_arr, image)
            
            counter += 1
            if counter >= max_training_set_size:
                break
    
    avg_pxl_arr /= counter
    avg_pxl_float = np.mean(avg_pxl_arr)


    G = GANNet(gen_layers, nn.LeakyReLU(), nn.Tanh(), drop_prob=0.0)
    D = GANNet(disc_layers, nn.LeakyReLU(), nn.Sigmoid(), drop_prob=0.1)
<<<<<<< Updated upstream
    train(G, D, image_list, avg_pxl_arr, avg_pxl_float, image_side_length, batch_size = 100)
    evaluate_finished_model(G, avg_pxl_arr, avg_pxl_float)
=======
    train(G, D, image_list, avg_pxl, image_side_length, batch_size = 100)
>>>>>>> Stashed changes

if __name__ == "__main__":
    main() 
