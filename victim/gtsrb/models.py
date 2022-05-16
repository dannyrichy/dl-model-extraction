import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from victim.scheduler import WarmupCosineLR

all_classifiers = {
    # "vgg11_bn": vgg11_bn(),
    # "vgg13_bn": vgg13_bn(),
    # "vgg16_bn": vgg16_bn(),
    # "vgg19_bn": vgg19_bn(),
    # "resnet18": resnet18(),
    # "resnet34": resnet34(),
    # "resnet50": resnet50(),
    # "densenet121": densenet121(),
    # "densenet161": densenet161(),
    # "densenet169": densenet169(),
    # "mobilenet_v2": mobilenet_v2(),
    # "googlenet": googlenet(),
}


class GTSRBModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # self.hparams = hparams
        self.lr = hparams.learning_rate
        self.weight_decay = hparams.weight_decay
        self.max_epochs = hparams.max_epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers[hparams.classifier]

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
