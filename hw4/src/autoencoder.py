import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        #TODO: 5%
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 5%
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.to(self.device)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)
        losses = []

        for epoch in tqdm(range(epochs), desc="Training Autoencoder"):
            total_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                outputs = self(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(data_loader)
            losses.append(epoch_loss)

        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Autoencoder Training Loss")
        plt.savefig("autoencoder_training_loss.png")
        plt.clf()
    
    def transform(self, X):
        #TODO: 2%
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        encoded = self.encoder(X_tensor)
        return encoded.cpu().detach().numpy()
    
    def reconstruct(self, X):
        #TODO: 2%
        # Note that the input X is the raw data, not the transformed data
        X_transformed = self.transform(X)
        X_transformed = torch.tensor(X_transformed, dtype=torch.float32).to(self.device)
        decoded = self.decoder(X_transformed)
        return decoded.cpu().detach().numpy()


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add_noise(self, x):
        #TODO: 3%
        # Note that torch.randn_like(x) samples from a normal distribution with mean 0 and variance 1
        noise = self.noise_factor * torch.randn_like(x)
        return x + noise
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.to(self.device)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)
        losses = []

        for epoch in tqdm(range(epochs), desc="Training Denoising Autoencoder"):
            total_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                noisy_batch = self.add_noise(batch)
                outputs = self(noisy_batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(data_loader)
            losses.append(epoch_loss)

        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Denoising Autoencoder Training Loss")
        plt.savefig("denoising_autoencoder_training_loss.png")
        plt.clf()
