
import matplotlib.pyplot as plt
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

def train(G, D, training_images, avg_pxl_arr, avg_pxl_float, image_side_length, batch_size):
    
    training_set_size = len(training_images)
    print("Training Set Size: " + str(training_set_size))

    dataloader = DataLoader(training_images, batch_size=batch_size, shuffle=True)

    D_learning_rate = 0.00008
    G_learning_rate = 0.00006

    max_epochs = 150
    loss = nn.BCELoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    G = G.to(device)
    D = D.to(device)

    D_optimizer = optim.Adam(D.parameters(), lr=D_learning_rate, betas=(0.0, 0.99))
    G_optimizer = optim.Adam(G.parameters(), lr=G_learning_rate, betas=(0.5, 0.99))

    epochs = []
    D_mean_true_losses = []
    D_mean_fake_losses = []
    G_mean_losses = []
    
    false_labels = torch.zeros((batch_size, 1), device=device)

    for epoch in range(max_epochs):

        epochs.append(epoch)

        D_epoch_mean_true_loss = 0.0
        D_epoch_mean_fake_loss = 0.0
        G_epoch_mean_loss = 0.0
        num_images_trained = 0

        for i, data in enumerate(dataloader):

            true_labels = 1-torch.abs(torch.randn((batch_size, 1), device=device)*0.05)
            
            # Generate fake image and sample true image
            # Does not need to include last few images outside of batch_size constraints
            true_data = (data.to(device)*2.0)-1.0
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

        print("Epoch " + '{:03}'.format(epoch)
                + ": DTL: " + '{:06.4f}'.format(D_mean_true_losses[-1])
                + ", DFL: " + '{:06.4f}'.format(D_mean_fake_losses[-1])
                + ", GL: " + '{:06.4f}'.format(G_mean_losses[-1]))

    # Plot the losses
    x = epochs 
    y1 = D_mean_true_losses 
    y2 = D_mean_fake_losses
    y3 = G_mean_losses
 
    plt.plot(x, y1, label = "Discriminator True Loss")
    plt.plot(x, y2, label = "Discriminator Fake Loss") 
    plt.plot(x, y3, label = "Generator Loss") 

    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.title('GAN Losses Over Time') 
    plt.legend()
  
    # Save plot
    plt.savefig("plots/GAN.jpg")

def evaluate_finished_model(G, avg_pxl_float, mean_real_img_devs):
    G.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    final_gen_imgs = 50
    noise = torch.randn(final_gen_imgs, G.layer_sizes[0], device=device)

    fake_data = G(noise).detach()
    counter = 0
    stan_dev = 0
    gen_img_devs = 0
    mean_gen_pxl_val = 0
    
    for i in fake_data:
        export_image(0.5*(i.numpy()+1.0), 'finished_' + str(counter))
        counter += 1
        # Calculates standard deviation of pixel values for each generated image
        f_arr = i.detach().cpu().numpy()
        gen_img_devs += np.std(f_arr)

        # Accumulates calculations for standard deviations comparing each generated image to average training image
        f_float = np.mean(f_arr)
        mean_gen_pxl_val += f_float
        stan_dev += np.subtract(f_float, avg_pxl_float)**2
    
    # Averages pixel value standard deviations across all generated images
    mean_gen_img_devs = gen_img_devs/final_gen_imgs
    mean_gen_pxl_val /= final_gen_imgs
   
    # Calculates standard deviation of fake images compared to average training image
    stan_dev /= final_gen_imgs
    stan_dev = np.sqrt(stan_dev)
    
    print("Standard deviation of final generated images' pixel values compared to the mean of training images' pixel values: " + '{:06.4f}'.format(stan_dev)
          + " \n Mean pixel value of generated images: " + '{:06.4f}'.format(mean_gen_pxl_val)
          + " \n Mean pixel value of training images: " + '{:06.4f}'.format(avg_pxl_float*2.0 - 1.0)
          + " \n Mean of standard deviations of pixel values across generated images: " + '{:06.4f}'.format(mean_gen_img_devs)
          + " \n Mean of standard deviations of pixel values across training images: " + '{:06.4f}'.format(mean_real_img_devs))

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
    # Possible choices: "dog", "cat", "corgi"
    animal_type = "cat"
    max_training_set_size = 99999
    global image_side_length
    image_side_length = 128
    gen_layers = [32, 512, 512, 512, image_side_length*image_side_length]
    disc_layers = [image_side_length*image_side_length, 512, 512, 256, 128, 64, 1]

    list_files = listdir(getcwd() + "/afhq/" + animal_type)

    list_files = natsorted(list_files)
    image_list = []
    avg_pxl_arr = np.zeros(image_side_length**2)
    img_devs = 0
    counter = 0

    for filename in list_files:
        if ".jpg" in filename:
            image = ski.io.imread(getcwd() + "/afhq/" + animal_type + "/" + filename, as_gray = True)
            # Scale image down to image_side_length
            image = ski.transform.rescale(image, image_side_length/512.0, anti_aliasing = True)
            # Flatten matrix to vector
            image = np.reshape(image, -1)
            image_list.append(image)
            
            # Adds pixelations values element-wise for all training images
            avg_pxl_arr = np.add(avg_pxl_arr, image)
            
            # Accumulates standard deviation of pixel values for each training image
            img_devs += np.std(image)
            
            counter += 1
            if counter >= max_training_set_size:
                break
    # Averages standard deviations across all training images
    mean_real_img_devs = img_devs/counter
    
    # Calculates average training image pixelation values element-wise in array of length image_side_length x image_side_length
    avg_pxl_arr /= counter
    
    # Calculates average pixelation value across all training images
    avg_pxl_float = np.mean(avg_pxl_arr)

    G = GANNet(gen_layers, nn.LeakyReLU(), nn.Tanh(), drop_prob=0.0)
    D = GANNet(disc_layers, nn.LeakyReLU(), nn.Sigmoid(), drop_prob=0.1)

    train(G, D, image_list, avg_pxl_arr, avg_pxl_float, image_side_length, batch_size = 40)
    evaluate_finished_model(G, avg_pxl_float, mean_real_img_devs)

if __name__ == "__main__":
    main() 
