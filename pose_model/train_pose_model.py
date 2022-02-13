import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from torchvision.datasets import ImageFolder


class PoseTrainModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr,
        image_size,
        model_arch,
    ):
        super().__init__()

        self.save_hyperparameters(
            "num_classes",
            "lr",
            "image_size",
            "model_arch",
        )

        assert self.hparams.model_arch in [
            "resnet18",
            "resnet34",
            "resnet50",
        ]

        model_arch = getattr(torchvision.models, self.hparams.model_arch)
        self.model = model_arch()

        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            self.hparams.num_classes
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_acc1 = torchmetrics.Accuracy(top_k=1)
        self.train_acc5 = torchmetrics.Accuracy(top_k=5)
        self.val_acc1 = torchmetrics.Accuracy(top_k=1)
        self.val_acc5 = torchmetrics.Accuracy(top_k=5)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
        )

        return optimizer

    def _stage_step(self, batch, key):
        X_batch, y_batch = batch
        out = self.model(X_batch)
        loss_batch = self.loss(out, y_batch)
        self.log(
            f"{key}_loss",
            loss_batch,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )

        if key == "train":
            acc1 = self.train_acc1
            acc5 = self.train_acc5
        else:
            acc1 = self.val_acc1
            acc5 = self.val_acc5

        out_probs = F.softmax(out, dim=1)
        self.log(
            f"{key}_acc_1",
            acc1(out_probs, y_batch),
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )
        self.log(
            f"{key}_acc_5",
            acc5(out_probs, y_batch),
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            prog_bar=True,
        )

        return loss_batch

    def training_step(self, batch, batch_idx):
        return self._stage_step(batch, key="train")

    def validation_step(self, batch, batch_idx):
        return self._stage_step(batch, key="val")


@hydra.main(config_path="config", config_name="train_pose_base.yaml")
def main(config):
    tb_logger = TensorBoardLogger("tb_logs", name="train_pose_tb")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        every_n_epochs=1,
        save_last=True,
        save_top_k=-1,
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        strategy=DDPPlugin(find_unused_parameters=True),
        **config.trainer,
    )

    train_augmentations = []
    if config.train_transforms.random_resize is True:
        resize_op = torchvision.transforms.RandomResizedCrop(
            size=(
                config.pose_train_module.image_size,
                config.pose_train_module.image_size,
            )
        )
    else:
        resize_op = torchvision.transforms.Resize(
            size=(
                config.pose_train_module.image_size,
                config.pose_train_module.image_size,
            )
        )
    train_augmentations.append(resize_op)
    if config.train_transforms.random_horizontal_flip is True:
        train_augmentations.append(
            torchvision.transforms.RandomHorizontalFlip(),
        )

    train_transform = torchvision.transforms.Compose(
        train_augmentations + [torchvision.transforms.ToTensor()]
    )

    val_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(
                    config.pose_train_module.image_size,
                    config.pose_train_module.image_size,
                )
            ),
            torchvision.transforms.ToTensor(),
        ]
    )
    train_dataset = ImageFolder(
        root=config.datamodule.train_dataset,
        transform=train_transform,
    )
    val_dataset = ImageFolder(
        root=config.datamodule.val_dataset,
        transform=val_transform,
    )
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.datamodule.batch_size,
        num_workers=config.datamodule.num_workers,
    )

    pose_train_module = PoseTrainModule(
        num_classes=len(train_dataset.classes),
        **config.pose_train_module,
    )

    trainer.fit(
        model=pose_train_module,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
