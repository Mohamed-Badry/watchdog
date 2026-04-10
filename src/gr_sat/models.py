import torch
import torch.nn as nn
import torch.nn.functional as F

class TelemetryVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Telemetry Anomaly Detection.
    Maps physical bounds into a probabilistic latent space.
    """
    def __init__(self, input_dim=5, hidden_dim=12, latent_dim=3):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)
        
    def reparameterize(self, mu, logvar):
        # Sample standard deviation
        std = torch.exp(0.5 * logvar)
        # Sample noise
        eps = torch.randn_like(std)
        # Reparameterization trick
        return mu + eps * std
        
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, kld_weight=0.01):
    """
    Loss function for VAE.
    1. MSE Loss (Reconstruction)
    2. KL Divergence (Latent space regularization)
    """
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Using 'mean' to keep it scaled nicely with MSE
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + kld_weight * KLD
