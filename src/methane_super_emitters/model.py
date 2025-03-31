import lightning as L
import torchmetrics
import torch
import torch.nn as nn


class SuperEmitterLocator(L.LightningModule):
    def __init__(self, fields):
        self.fields = fields
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(len(self.fields), 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 32),  # Output layer
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat_flat = y_hat.view(y.shape[0], -1)
        y_indices = y.view(y.shape[0], -1).argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(y_hat_flat, y_indices)
        self.log("train_loss", loss)
        return loss

class SuperEmitterDetector(L.LightningModule):
    def __init__(self, fields, dropout=0.4, weight_decay=0.01, lr=1e-3):
        super().__init__()
        self.fields = fields
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr

        self.conv_layers = nn.Sequential(
            nn.Conv2d(len(fields), 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = nn.functional.binary_cross_entropy(outputs, labels.float())
        acc = ((outputs > 0.5).int() == labels).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = nn.functional.binary_cross_entropy(outputs, labels.float())
        acc = ((outputs > 0.5).int() == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        acc = ((outputs > 0.5).int() == labels).float().mean()
        self.log("test_acc", acc)

