#!/usr/bin/env python

import os
import warnings
import logging
import argparse
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from functools import partial
import multiprocessing
import torch.multiprocessing
import torch
import torch.nn as nn
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, WeightedRandomSampler  # , random_split
from typing import Any
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC
from knockknock import slack_sender

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.simplefilter("ignore", category=UserWarning)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torchvision.dataset").setLevel(logging.ERROR)
N_CPU = multiprocessing.cpu_count()
# data location
log_dir = "/Users/jrudoler/Library/CloudStorage/Box-Box/JR_CML/pytorch_logs/"


class LitPrecondition(pl.LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate, weight_decay, batch_size):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.condition = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=2 * input_dim,
                kernel_size=2,
                padding=1,
                groups=input_dim,
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Conv1d(
                in_channels=2 * input_dim,
                out_channels=2 * input_dim,
                kernel_size=4,
                padding=1,
                groups=2 * input_dim,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(
                in_channels=2 * input_dim,
                out_channels=input_dim,
                kernel_size=4,
                padding=1,
                groups=input_dim,
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Flatten(),
        )
        self.logistic = nn.Sequential(nn.Linear(output_dim, 1, bias=True), nn.Sigmoid())
        self.save_hyperparameters()

    def forward(self, X):
        X_cond = self.condition(X)
        probs = self.logistic(X_cond)
        return probs

    def training_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.float(), y.float()
        y_hat = torch.squeeze(self.forward(X))
        loss = F.binary_cross_entropy(y_hat, y)
        auroc = BinaryAUROC()
        train_auc = auroc(y_hat, y)
        self.log_dict(
            {"Loss/train": loss, "AUC/train": train_auc},
            on_epoch=False,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.float(), y.float()
        y_hat = torch.squeeze(self.forward(X))
        loss = F.binary_cross_entropy(y_hat, y)
        auroc = BinaryAUROC()
        test_auc = auroc(y_hat, y)
        self.log_dict(
            {"Loss/val": loss, "AUC/val": test_auc},
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.float(), y.float()
        y_hat = torch.squeeze(self.forward(X))
        loss = F.binary_cross_entropy(y_hat, y)
        auroc = BinaryAUROC()
        test_auc = auroc(y_hat, y)
        self.log_dict(
            {"Loss/test": loss, "AUC/test": test_auc},
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
        )

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                threshold=1e-4,
                threshold_mode="rel",
                patience=10,
                verbose=False,
            ),
            # The unit of the scheduler's step size, 'epoch' or 'step'.
            "interval": "epoch",
            # How many epochs/steps pass between calling scheduler.step()
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "Loss/train",
            # If `True`, will enforce that value 'monitor' is available
            "strict": True,
            # custom name if using the `LearningRateMonitor` callback
            "name": "learning_rate",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class ltpFR2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        subject,
        train_sess,
        holdout_sess,
        data_dir: str = "./",
        batch_size: int = 128,
        across="session",
    ):
        super().__init__()
        self.subject = subject
        self.train_sess = train_sess
        self.holdout_sess = holdout_sess
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.across = across

    def setup(self, stage: str):
        if stage == "fit":
            self.train_file_crit = (
                lambda s: s.endswith(".pt")
                and (self.subject in s)
                and any([f"sess_{sess}_" in s for sess in self.train_sess])
            )
            self.train_dataset = DatasetFolder(
                self.data_dir,
                loader=partial(torch.load),
                is_valid_file=self.train_file_crit,
            )
            print("length of train dataset:", len(self.train_dataset))
            self.n_features = self.train_dataset[0][0].shape[0]

            self.test_file_crit = (
                lambda s: s.endswith(".pt")
                and (self.subject in s)
                and any([f"sess_{sess}_" in s for sess in self.holdout_sess])
            )
            self.test_dataset = DatasetFolder(
                self.data_dir,
                loader=partial(torch.load),
                is_valid_file=self.test_file_crit,
            )
            print("length of test dataset:", len(self.test_dataset))

    def train_dataloader(self):
        cls_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_dataset.targets),
            y=self.train_dataset.targets,
        )
        weights = cls_weights[self.train_dataset.targets]
        sampler = WeightedRandomSampler(
            weights, len(self.train_dataset), replacement=True  # type: ignore
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=N_CPU,
            prefetch_factor=10,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            pin_memory=True,
        )


@slack_sender(
    "https://hooks.slack.com/services/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",
    "D03SCTEJ0JJ",
)
def train_model(
    initial_model,
    subject,
    n_sess=24,
    holdout_sess=[5, 10, 15, 20],
    data_dir="/Users/jrudoler/data/scalp_features/",
    fast_dev_run: bool | int = False,
    seed=56,
    learning_rate=1e-2,
    weight_decay=0.5,
    batch_size=512,
    timestr=None,
):
    timestr = timestr or time.strftime("%Y%m%d-%H%M%S")
    _ = pl.seed_everything(seed, workers=True)

    train_sess = list(set(range(n_sess)) - set(holdout_sess))
    print("holdout: ", holdout_sess, "train: ", train_sess)
    # data module
    dm = ltpFR2DataModule(
        subject=subject,
        train_sess=train_sess,
        holdout_sess=holdout_sess,
        data_dir=data_dir,
        batch_size=batch_size,
    )
    dm.setup("fit")
    # create model
    model = torch.load(initial_model)
    model.hparams["learning_rate"] = learning_rate
    model.hparams["weight_decay"] = weight_decay
    model.hparams["batch_size"] = batch_size

    es = EarlyStopping(
        "Loss/val",
        min_delta=1e-3,
        patience=25,
        mode="min",
    )
    lr_mtr = LearningRateMonitor("epoch")
    check = ModelCheckpoint(monitor="Loss/train", mode="min")
    run_dir = f"run_tune_{subject}_{timestr}"
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="precondition_tune",
        version=run_dir,
        default_hp_metric=True,
    )
    logger.log_hyperparams(model.hparams)
    trainer = Trainer(
        min_epochs=50,
        max_epochs=300,
        limit_val_batches=10,
        accelerator="mps",
        devices=1,
        callbacks=[lr_mtr, es, check],
        logger=logger,
        log_every_n_steps=10,
        fast_dev_run=fast_dev_run,
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)
    if fast_dev_run:
        return
    model = LitPrecondition.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path  # type: ignore
    )  # Load best checkpoint after training
    torch.save(model, log_dir + f"models/tuned_model_{subject}_{timestr}.pt")
    test_result = trainer.test(model=model, datamodule=dm)
    result_df = pd.DataFrame(test_result)
    result_df.to_csv(
        log_dir
        + f"test_results/precond_test_{subject}_{weight_decay:.3e}_{timestr}.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with given parameters.")
    parser.add_argument("-m", "--model_path_init", help="pretrained model")
    parser.add_argument("-S", "--subject", nargs="+", help="subject(s) to fine tune")
    parser.add_argument(
        "--holdout",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="holdout sessions",
    )
    parser.add_argument(
        "-n", "--n_sess", type=int, default=24, help="number of sessions"
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        default="/Users/jrudoler/data/scalp_features/",
        help="path to data directory",
    )
    parser.add_argument(
        "-f",
        "--fast_dev_run",
        type=lambda x: int(x) if x.isdigit() else x.lower() == "true",
        default=False,
        help="fast development run (bool or int)",
    )
    parser.add_argument("-s", "--seed", type=int, default=56, help="random seed")
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=1e-2, help="learning rate"
    )
    parser.add_argument(
        "-w", "--weight_decay", type=float, default=0.5, help="weight decay"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="batch size")

    args = parser.parse_args()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    models_dir = log_dir + f"models/tuned_models_{timestr}/"
    os.makedirs(models_dir)
    start = time.time()
    print(args.data_dir)
    print(args.subject)
    for subject in args.subject:
        train_model(
            initial_model=args.model_path_init,
            subject=subject,
            holdout_sess=args.holdout,
            n_sess=args.n_sess,
            data_dir=args.data_dir,
            fast_dev_run=args.fast_dev_run,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            timestr=timestr,
        )
    end = time.time()
    print(f"Done! (runtime: {timedelta(seconds=end-start)})")
