from typing import Optional, Tuple, List, Callable, Dict, Any
import os
from glob import glob

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import albumentations as A
import cv2

import pli_transforms.augmentations as aug
from pli_transforms.augmentations.pytorch import ToTensorPLI

from pli_cyto import utils
from pli_cyto.datamodules.components.sections import SectionSampler

# Distributed
from atlasmpi import MPI

comm = MPI.COMM_WORLD

log = utils.get_logger(__name__)


class AlignedDataModule(LightningDataModule):

    def __init__(
            self,
            cyto_files: [str, List[str]],
            trans_files: [str, List[str]],
            dir_files: [str, List[str]],
            ret_files: [str, List[str]],
            train_sections: List[int],
            val_sections: List[int],
            train_size: int = 2 ** 10,
            val_size: int = 128,
            val_seed: int = 299792458,
            batch_size: int = 64,
            cyto_patch_size: Tuple[int, int] = (1024, 1024),
            pli_patch_size: Tuple[int, int] = (256, 256),
            channels: int = 1,
            num_workers: int = 0,
            sections_to_ram: bool = True,
            split_driver: bool = False,
            pin_memory: bool = False,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=True)

        # data transformations
        self.train_transforms = A.Compose([
            aug.RandomRotatePLI(
                always_apply=True,
                limit=(-180, 180),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
            ),
            aug.RandomDirectionOffsetPLI(),
            aug.RandomFlipPLI(),
            aug.ScaleThicknessPLI(
                log_range=(-1.0, 1.0),
                trans_max=1.0,
                clip_max=1.5,
                always_apply=True
            ),
            aug.ScaleAttenuationPLI(
                log_range=(-1.0, 1.0),
                trans_max=1.0,
                clip_max=1.5,
                always_apply=True 
            ),
            aug.BlurPLI(
                blur_limit=(3, 5),
                sigma_limit=(0, 1.5),
                p=0.5),
            ToTensorPLI()
        ])

        self.train_sampler: Optional[Dataset] = None
        self.val_sampler: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if stage in ['fit', None]:
            if not self.train_sampler or not self.val_sampler:
                if type(self.hparams.cyto_files) is str:
                    self.hparams.cyto_files = sorted(glob(self.hparams.cyto_files))
                if type(self.hparams.trans_files) is str:
                    self.hparams.trans_files = sorted(glob(self.hparams.trans_files))
                if type(self.hparams.dir_files) is str:
                    self.hparams.dir_files = sorted(glob(self.hparams.dir_files))
                if type(self.hparams.ret_files) is str:
                    self.hparams.ret_files = sorted(glob(self.hparams.ret_files))

                self.train_sampler = SectionSampler(
                    [self.hparams.cyto_files[i] for i in self.hparams.train_sections],
                    [self.hparams.trans_files[i] for i in self.hparams.train_sections],
                    [self.hparams.dir_files[i] for i in self.hparams.train_sections],
                    [self.hparams.ret_files[i] for i in self.hparams.train_sections],
                    cyto_patch_size=self.hparams.cyto_patch_size,
                    pli_patch_size=self.hparams.pli_patch_size,
                    transform=self.train_transforms,
                    n_samples=self.hparams.train_size,
                    ram=self.hparams.sections_to_ram,
                    driver=('split' if self.hparams.split_driver else None)
                )

                self.val_sampler = SectionSampler(
                    [self.hparams.cyto_files[i] for i in self.hparams.val_sections],
                    [self.hparams.trans_files[i] for i in self.hparams.val_sections],
                    [self.hparams.dir_files[i] for i in self.hparams.val_sections],
                    [self.hparams.ret_files[i] for i in self.hparams.val_sections],
                    cyto_patch_size=self.hparams.cyto_patch_size,
                    pli_patch_size=self.hparams.pli_patch_size,
                    transform=self.train_transforms,
                    n_samples=self.hparams.val_size,
                    seed=self.hparams.val_seed,
                    ram=self.hparams.sections_to_ram,
                    driver=('split' if self.hparams.split_driver else None)
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_sampler,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_sampler,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
