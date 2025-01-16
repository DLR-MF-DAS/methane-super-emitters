import lightning as L
import torchmetrics
import torch
import torch.nn as nn

class SuperEmitterDetector(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.cnv = nn.Conv2d(1, 128, 5, 4)
        self.rel = nn.ReLU()
        self.bn = nn.BatchNorm2d(128)
        self.mxpool = nn.MaxPool2d(4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

    def forward(self, x):
        out = self.bn(self.rel(self.cnv(x)))
        out = self.flat(self.mxpool(out))
        out = self.rel(self.fc1(out))
        out = self.rel(self.fc2(out))
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        img = x.view(-1, 1, 32, 32)
        label = y.view(-1)
        out = self(img)
        loss = torch.nn.functional.binary_cross_entropy(out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        img = x.view(-1, 1, 32, 32)
        label = y.view(-1)
        out = self(img)
        loss = torch.nn.functional.binary_cross_entropy(out, y)
        #out = nn.Softmax(-1)(out)
        #logits = torch.argmax(out, dim=1)
        out = torch.where(out > 0.5, 1.0, 0.0)
        accu = self.accuracy(out, y)
        self.log('accuracy', accu)
        return loss, accu
        
