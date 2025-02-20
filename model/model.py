import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2)  # Output z_mean and z_log_var
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x) #[BS=32, latent_dim*2=128]
        z_mean = h[:, :self.latent_dim] #[BS=32, latent_dim=32]
        z_log_var = h[:, self.latent_dim:] #[BS=32, latent_dim=32]
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var) # take std from log_var --> sqrt(exp(log_var))
        # reparameterization trick, since sampling torch.normal(z_mean, std) is not differentiable
        eps = torch.randn_like(std) 
        return z_mean + eps * std

    def decode(self, z):
        return self.decoder(z) #[BS=32, n_data=5=x,y,rgb]

    def sample_new_img_from_rand(self, num_samples=5000):
        """
        Sample new data points from latent space: random Gaussian data
        Params:
            num_samples: Number of data points to sample
        Return:
            generated_points: Generated data points in numpy. Shape=[num_samples, xyrgb=5]
        """ 
        z_samples = torch.randn(num_samples, self.latent_dim) #[num_samples, e], sampling from N(0,1)
        z_samples = z_samples.to(self.decoder[0].weight.device)
        generated_points = self.decode(z_samples).detach().cpu().numpy()
        return generated_points
    
    def sample_new_img_from_learned_latent(self, input=None):
        """
        Sample new data points from latent space: learned mean and var latent data. In this case, we use the original data input to derive the latent z_mean, z_log_var
        Params:
            input: The original data input used to derive z_mean and z_log_var. Shape=[num_samples, xyrgb=5]
        Return:
            generated_points: Generated data points in numpy. Shape=[num_samples, xyrgb=5]
        """ 
        assert input is not None, "input cannot be None!"

        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        input = input.to(self.decoder[0].weight.device)
        
        z_mean, z_log_var = self.encode(input)
        z = self.reparameterize(z_mean, z_log_var)
        generated_points = self.decode(z).detach().cpu().numpy()
        return generated_points

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_mean, z_log_var
    
def loss_function(x, x_reconstructed, z_mean, z_log_var, beta=1.0):
    """
    VAE loss, consisting of reconstruction loss via MSE and aligment loss via KL-div
    Note:
    1. For kl_loss, it is computed between latent z (Gaussian) and unit Gaussian. The derivation proof can be seen here:
       https://daffy-slash-cc0.notion.site/Non-GAN-Density-Estimation-1bc6e95f77354c7392f72c40470372ed 
    """
    reconstruction_loss = F.mse_loss(x_reconstructed, x)/len(z_mean) # scalar, e.g., 2.5e8
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var))/len(z_mean) # scalar, e.g., 3.9e13
    loss = reconstruction_loss + beta*kl_loss
    return loss, reconstruction_loss.item(), kl_loss.item()