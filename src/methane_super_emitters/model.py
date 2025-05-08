import lightning as L
import torchmetrics
import torch
import torch.nn as nn
import torch.optim as optim

class SuperEmitterDetector(L.LightningModule):
    def __init__(self, fields, dropout=0.6, weight_decay=2e-3, lr=1e-3):
        super().__init__()
        self.fields = fields
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr

        self.conv_layers = nn.Sequential(
            nn.Conv2d(len(fields), 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.squeeze()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _shared_step(self, batch, stage):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels.float())

        acc = ((torch.sigmoid(outputs) > 0.5).int() == labels).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping to prevent overfitting by controlling weight explosion."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
