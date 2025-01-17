import os
from typing import Any, List, Dict, Callable, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torchvision.transforms.functional as T
from pytorch_lightning import LightningModule
import hydra

from pli_transforms.geometric import rot90_mods_tensor
from pli_styles.modality.inclination import corr_factor_linear, inclination_from_retardation
from pli_styles.modality.fom import hsv_fom
from ffreg import EulerRegistration

import pli_cyto.utils.functions as f
import pli_cyto.utils.loss as losses

# Distributed
from atlasmpi import MPI

comm = MPI.COMM_WORLD


class DFTStyleLitModule(LightningModule):

    def __init__(
            self,
            gen: Dict,
            style_net: Dict,
            out_channels: int = 1,
            lr: float = 0.001,
            angles: List[float] = [0.0],
            cyto_patch_size: Tuple[int, int] = [512, 512],
            pli_patch_size: Tuple[int, int] = [256, 256],
            min_overlap: float = 0.25,
            keep_best: int = -1,
            rec_loss: str = 'mae_loss',
            style_loss: str = 'mse_loss',
            equivariant_loss: str = 'mse_loss',
            layer_weights: List[float] = [1.0],
            rec_loss_weight: float = 1.0,
            style_loss_weight: float = 1.0,
            equivariant_loss_weight: float = 0.0,
            register_target: bool = True,
            registration_method: str = 'mse',
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        if self.hparams.register_target:
            resample_shape = cyto_patch_size
        else:
            resample_shape = pli_patch_size

        if registration_method.lower() == 'pc_hann':
            registration_method = 'pc'
            window_type = 1
        elif registration_method.lower() == 'bipc_hann':
            registration_method = 'bipc'
            window_type = 1
        else:
            window_type = 0

        self.euler_reg = EulerRegistration(
            method=registration_method,
            circular=False,
            min_overlap=min_overlap,
            batch_parallel=False,
            implementation='exhaustive',
            resample_shape=resample_shape,
            rotation_angles=angles,
            resample_mode='bilinear',
            reduce_channels=True,
            window_type=window_type,
        )

        # used models
        self.gen = hydra.utils.instantiate(gen)

        # used style model
        self.style_net = hydra.utils.instantiate(style_net)

        # loss function
        self.rec_loss = getattr(losses, rec_loss)
        self.equivariant_loss = getattr(losses, equivariant_loss)


    def forward(self, trans: Tensor, dir: Tensor, ret: Tensor):
        return self.gen(trans=trans, dir=dir, ret=ret)

    def gen_step(self, batch: Any):
        fake = self.forward(batch['trans'], batch['dir'], batch['ret'])
        real = batch['cyto']

        if self.hparams.register_target:
            # Perform registration of (moving) real target to the (fixed) fake prediction
            with torch.no_grad():
                best_real, best_mask, _, scores, _ = self.euler_reg.forward(
                    fake.detach(),
                    real,
                    return_score_map=False,
                )

            best_fake = fake
        else:
            # Perform registration of (moving) fake prediction to the (fixed) real target
            best_fake, best_mask, _, scores, _ = self.euler_reg.forward(
                real,
                fake,
                return_score_map=False,
            )
            best_real = real

        ## Equivariant Loss ##

        if self.hparams.equivariant_loss_weight > 0.:
            trans_180, dir_180, ret_180 = rot90_mods_tensor(batch['trans'], batch['dir'], batch['ret'], 2)
            with torch.no_grad():
                fake_180 = self.forward(trans_180, dir_180, ret_180)
                fake_equivariant = torch.rot90(fake_180, -2, dims=(-2, -1))
            equivariant_loss = self.equivariant_loss(fake_equivariant, fake)
        else:
            equivariant_loss = 0.

        ## Reduce Best Fits ##

        if 0 <= self.hparams.keep_best < len(scores):
            best_ix = torch.argsort(scores)[:self.hparams.keep_best]
        else:
            best_ix = ...

        target = best_real[best_ix]
        pred = best_fake[best_ix]
        masks = best_mask[best_ix]

        ## Style Loss ##

        if self.hparams.style_loss_weight > 0.:
            if self.hparams.style_loss.lower() == 'mse_loss':
                # Old implementation of style loss
                real_crop = T.center_crop(real, fake.shape[-2:])
                style_loss = losses.style_loss(fake, real_crop, self.style_net, self.hparams.layer_weights, F.mse_loss)
            elif self.hparams.style_loss.lower() == 'masked_gram':
                style_loss = losses.style_loss(pred, target, self.style_net, self.hparams.layer_weights, F.mse_loss, masks)
            else:
                style_loss = 0.
        else:
            style_loss = 0.

        ## Reconstruction Loss ##

        if self.hparams.rec_loss_weight > 0.:
            rec_loss = self.rec_loss(target, pred, masks)
        else:
            rec_loss = 0.

        return style_loss, rec_loss, equivariant_loss, best_fake, best_real, best_mask

    def training_step(self, batch: Any, batch_idx: int):
        # Train the generator

        style_loss, rec_loss, equivariant_loss, _, _, _ = self.gen_step(batch)
        if self.hparams.rec_loss_weight > 0.:
            self.log("train/rec_loss", rec_loss.item())
        if self.hparams.style_loss_weight > 0.:
            self.log("train/style_loss", style_loss.item())
        if self.hparams.equivariant_loss_weight > 0.:
            self.log("train/equivariant_loss", equivariant_loss.item())

        total_loss = self.hparams.style_loss_weight * style_loss + \
                     self.hparams.rec_loss_weight * rec_loss + \
                     self.hparams.equivariant_loss_weight * equivariant_loss
        self.log("train/loss", total_loss.item())

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": total_loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        style_loss, rec_loss, equivariant_loss, best_fake, best_real, best_mask = self.gen_step(batch)

        if self.hparams.rec_loss_weight > 0.:
            self.log("val/rec_loss", rec_loss.item(), sync_dist=True)
        if self.hparams.style_loss_weight > 0.:
            self.log("val/style_loss", style_loss.item(), sync_dist=True)
        if self.hparams.equivariant_loss_weight > 0.:
            self.log("val/equivariant_loss", equivariant_loss.item(), sync_dist=True)

        total_loss = self.hparams.style_loss_weight * style_loss + \
                     self.hparams.rec_loss_weight * rec_loss + \
                     self.hparams.equivariant_loss_weight * equivariant_loss
        self.log("val/loss", total_loss.item(), sync_dist=True)

        out_dict = {'loss': total_loss, 'fake': best_fake.detach().cpu(), 'real': best_real.detach().cpu(),
                    'mask': best_mask.detach().cpu(), **dict((k, v.detach().cpu()) for k, v in batch.items())}

        return out_dict

    def validation_epoch_end(self, outputs: List[Any]):
        vis_max = 32
        im_path = "images"

        if comm.Get_rank() == 0:
            if not os.path.exists(im_path):
                os.mkdir(im_path)
            n = min(len(outputs[0]['trans']), vis_max)
            ncols = math.ceil(math.sqrt(n))
            nrows = math.ceil(n / ncols)

            if self.current_epoch == 0:
                trans_grid = make_grid(list(torch.clip(outputs[0]['trans'][:vis_max].cpu(), 0, 1)), nrow=nrows, value_range=(0, 1))
                save_image(trans_grid, f'{im_path}/transmittance.png')

                dir_grid = make_grid(list(outputs[0]['dir'][:vis_max].cpu() / math.pi), nrow=nrows, value_range=(0, 1))
                save_image(dir_grid, f'{im_path}/direction.png')

                ret_grid = make_grid(list(outputs[0]['ret'][:vis_max].cpu()), nrow=nrows, value_range=(0, 1))
                save_image(ret_grid, f'{im_path}/retardation.png')

                fom_list = []
                for t, d, r in zip(outputs[0]['trans'][:vis_max].cpu(), outputs[0]['dir'][:vis_max].cpu(), outputs[0]['ret'][:vis_max].cpu()):
                    t_numpy = f.array(t)[..., 0]
                    d_numpy = f.array(d)[..., 0]
                    r_numpy = f.array(r)[..., 0]
                    
                    corr = corr_factor_linear(t_numpy)
                    i_numpy = inclination_from_retardation(r_numpy, corr)
                    fom = hsv_fom(np.rad2deg(d_numpy), i_numpy)
                    fom = fom / 255.
                    fom_list.append(f.tensor(fom, dtype=torch.float32))

                fom_grid = make_grid(fom_list, nrow=nrows, value_range=(0, 1))
                save_image(fom_grid, f'{im_path}/fom.png')

                cyto_grid = make_grid(list(outputs[0]['cyto'][:vis_max].cpu()), nrow=nrows, value_range=(0, 1))
                save_image(cyto_grid, f'{im_path}/cresyl_original.png')

            fake_grid = make_grid(list(outputs[0]['fake'][:vis_max].cpu()), nrow=nrows, value_range=(0, 1))
            save_image(fake_grid, f'{im_path}/cresyl_fake_{self.current_epoch:04d}.png')

            real_values = outputs[0]['real'][:vis_max].cpu()
            real_values[~outputs[0]['mask'][:vis_max].cpu()] = 0
            real_grid = make_grid(list(real_values), nrow=nrows, value_range=(0, 1))
            save_image(real_grid, f'{im_path}/cresyl_real_registered_{self.current_epoch:04d}.png')

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
            Normally you'd need one. But in the case of GANs or similar you might have multiple.

            See examples here:
                https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
            """
        optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.lr)
        return optimizer
