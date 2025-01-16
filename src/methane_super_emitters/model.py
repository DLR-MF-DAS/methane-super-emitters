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
        self.fc3 = nn.Linear(64, 2)
        self.softmax = nn.Softmax()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()

    def forward(self, x):
        out = self.bn(self.rel(self.cnv(x)))
        out = self.flat(self.mxpool(out))
        out = self.rel(self.fc1(out))
        out = self.rel(self.fc2(out))
        out = self.fc3(out)
        return out

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, 2), target)

    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        img = x.view(-1, 1, 32, 32)
        label = y.view(-1)
        out = self(img)
        loss = self.loss_fn(out, label)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        img = x.view(-1, 1, 32, 32)
        label = y.view(-1)
        out = self(img)
        loss = self.loss_fn(out, label)
        out = nn.Softmax(-1)(out)
        logits = torch.argmax(out, dim=1)
        accu = self.accuracy(logits, label)
        self.log('accuracy', accu)
        return loss, accu
        
