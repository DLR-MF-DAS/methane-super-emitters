import lightning as L
import torchmetrics
import torch
import torch.nn as nn

class SuperEmitterDetector(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return torch.sigmoid(out)

    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=0.05)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = nn.functional.binary_cross_entropy(outputs, labels.float())
        acc = ((outputs > 0.5).int() == labels).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = nn.functional.binary_cross_entropy(outputs, labels.float())
        acc = ((outputs > 0.5).int() == labels).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        acc = ((outputs > 0.5).int() == labels).float().mean()
        self.log('test_acc', acc)
