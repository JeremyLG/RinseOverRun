import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
train = pd.read_csv("data/processed/train.csv").drop(["process_id", "phase_",
                                                      "final_rinse_total_turbidity_liter"], axis=1)
train = train[~train.isnull().any(axis=1)]
np_train = mms.fit_transform(train)
test = pd.read_csv("data/processed/valid.csv").drop(["process_id", "phase_",
                                                     "final_rinse_total_turbidity_liter"], axis=1)
test = test[~test.isnull().any(axis=1)]
np_test = mms.transform(test)
np_train.shape[1]
np_test.shape


class VAE(nn.Module):
    def __init__(self, shape):
        super(VAE, self).__init__()
        self.n_features = shape

        self.fc1 = nn.Linear(self.n_features, 50)
        self.fc21 = nn.Linear(50, 20)
        self.fc22 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 50)
        self.fc4 = nn.Linear(50, self.n_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.n_features))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def latent(self, x):
        mu, logvar = self.encode(x.view(-1, self.n_features))
        return self.reparameterize(mu, logvar)


reconstruction_function = nn.MSELoss(reduction="sum")


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, np_train.shape[1]), reduction='sum')
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


device = torch.device("cpu")
vae = VAE(np_train.shape[1]).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
tensor = torch.Tensor(np_train)
train_loader = DataLoader(tensor, batch_size=256, shuffle=True, num_workers=4)
tensor = torch.Tensor(np_test)
test_loader = DataLoader(tensor, batch_size=256, shuffle=True, num_workers=4)


def train_vae(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test_vae(epoch):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = vae(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


tr_l = []
te_l = []
for epoch in range(1, 10):
        tr_l.append(train_vae(epoch))
        te_l.append(test_vae(epoch))

df = pd.DataFrame(data={"train_loss": tr_l, "test_loss": te_l})
df.plot()

vae.latent(torch.Tensor(np_test[1]))


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = VAE()
    model.load_state_dict(torch.load(path))
    return model


save_model(vae, 'models/vae.pt')
new_vae = load_model('models/vae.pt')
new_vae.eval()
