import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import Dataset


# Define the model architecture using PyTorch Lightning
class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, lr=0.001):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lr = lr

    def forward(self, x):
        out = self.fc1(x.float())
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = nn.CrossEntropyLoss()(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = nn.CrossEntropyLoss()(out, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = nn.CrossEntropyLoss()(out, y)
        self.log("test_loss", loss)
        f1 = f1_score(y.cpu(), out.argmax(1).cpu(), average="macro")
        self.log("test_f1", f1)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


class NNDataset(Dataset):
    def __init__(self, X, y):
        # Add your dataset initialization code here
        self.X = X
        self.y = y

    def __len__(self):
        # Return the total number of samples in your dataset
        return len(self.X)

    def __getitem__(self, index):
        # Return a single sample from your dataset based on the given index
        X = torch.tensor(self.X[index], dtype=torch.long)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return X, y
