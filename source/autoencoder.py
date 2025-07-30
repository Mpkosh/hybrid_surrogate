import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    pass


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

        self.enc_mu = nn.Linear(hidden_size, latent_size)
        self.enc_logvar = nn.Linear(hidden_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.max_logvar = nn.Parameter(torch.ones(
            latent_size) * 0.5, requires_grad=True)
        self.min_logvar = nn.Parameter(torch.ones(
            latent_size) * -0.5, requires_grad=True)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x)
        log_var = self.enc_logvar(x)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
