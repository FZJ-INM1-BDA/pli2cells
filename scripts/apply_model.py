import click
import os

import numpy as np

from torch.utils.data import DataLoader
import torch

from pli_cyto.models.dft_style import DFTStyleLitModule
from pli_cyto.models.dft_wgan import DFTGanLitModule
from pli_cyto.utils.files import get_files, save_section, create_cyto
from pli_cyto.datamodules.components.sections import SectionDataset

# Distributed
from atlasmpi import MPI

comm = MPI.COMM_WORLD


@click.command()
@click.option("--ckpt", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--trans", type=str)
@click.option("--dir", type=str)
@click.option("--ret", type=str)
@click.option("--out", type=click.Path(file_okay=True, dir_okay=True))
@click.option("--norm_trans", type=float, default=None)
@click.option("--num_workers", type=int, default=0)
@click.option("--batch_size", type=int, default=1)
@click.option("--patch_size", type=int, default=256)
@click.option("--out_size", type=int, default=128)
@click.option("--name", type=str, default="Cresyl")
@click.option("--ram", default=False, is_flag=True)
@click.option("--rotate180", default=False, is_flag=True)
def cli(ckpt, trans, dir, ret, out, norm_trans, num_workers, batch_size, patch_size, out_size, ram, name, rotate180):
    rank = comm.Get_rank()
    size = comm.size

    if torch.cuda.is_available():
        available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        print(f"Found {len(available_gpus)} GPUs")
        device = available_gpus[rank % len(available_gpus)]
    else:
        device = 'cpu'
    print(f"Use device {device} on rank {rank}")

    # Create model
    try:
        model = DFTStyleLitModule.load_from_checkpoint(ckpt)
    except:
        model = DFTGanLitModule.load_from_checkpoint(ckpt)
    model.to(device)
    print(f"Model loaded on rank {rank}")
    
    if not os.path.exists(out):
        os.makedirs(out, exist_ok=True)

    for trans_file, dir_file, ret_file, out_file in get_files(trans, dir, ret, out, rank, out_name=name):
        print(f"Initialize DataLoader for {trans_file}, {dir_file}, {ret_file}")
        patch_shape = [patch_size, patch_size]
        out_shape = [out_size, out_size]
        section_dataset = SectionDataset(
            trans_file=trans_file,
            dir_file=dir_file,
            ret_file=ret_file,
            patch_shape=patch_shape,
            out_shape=out_shape,
            ram=ram,
            norm_trans=norm_trans
        )
        section_loader = DataLoader(
            section_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        chanels = model.hparams.out_channels

        out_section = create_cyto(model, section_loader, rank, channels=chanels, dtype=np.uint8, rotate180=rotate180)

        spacing = section_dataset.trans_section_mod.spacing
        origin = section_dataset.trans_section_mod.origin
        save_section(
            out_section,
            out_file,
            spacing=spacing,
            origin=origin,
            modality="Cresyl",
            maximum=255,
            minimum=0
        )


if __name__ == '__main__':
    cli()
