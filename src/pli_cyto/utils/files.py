import os
import re
from typing import List, Tuple, Any
from glob import glob

import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from torchvision.transforms import functional as T
from torch.utils.data import DataLoader
import torch

from plio import Section
from pli_transforms.geometric import rot90_mods

import pli_cyto.utils.functions as f


def get_files(trans, dir, ret, out, rank=0, size=1, out_name="Cresyl"):
    trans_files = sorted(glob(trans))
    dir_files = sorted(glob(dir))
    ret_files = sorted(glob(ret))

    if os.path.isdir(out):
        out_files = []
        for d_f in dir_files:
            d_fname = os.path.basename(d_f)
            d_base = os.path.splitext(d_fname)[0]
            out_file = re.sub("direction", out_name, d_base, flags=re.IGNORECASE)
            if out_name not in out_file:
                out_file += f"_{out_name}.h5"
            else:
                out_file += ".h5"
            out_files.append(os.path.join(out, out_file))
    else:
        out_files = [out]

    for i, (trans_file, dir_file, ret_file, out_file) \
            in enumerate(zip(trans_files, dir_files, ret_files, out_files)):
        if i % size == rank:
            if not os.path.isfile(out_file):
                yield trans_file, dir_file, ret_file, out_file
            else:
                print(f"{out_file} already exists. Skip.")


def create_cyto(
        gen_model: Any,
        section_loader: DataLoader,
        rank: int,
        channels: int = 1,
        dtype: int = np.uint8,
        rotate180: bool = False,
):
    out_shape = section_loader.dataset.out_shape

    print("Initialize output section...")
    if channels == 1:
        section_shape = section_loader.dataset.trans_section.shape
    else:
        section_shape = (*section_loader.dataset.trans_section.shape, channels)
    out_section = np.zeros(section_shape, dtype=dtype)

    def get_outputs(batch, network):
        if rotate180:
            trans, dir, ret = rot90_mods(batch['trans'], batch['dir'], batch['ret'], 2)
            trans = torch.tensor(trans)
            dir = torch.tensor(dir)
            ret = torch.tensor(ret)
        else:
            trans, dir, ret = batch['trans'], batch['dir'], batch['ret']
        with torch.no_grad():
            network.eval()
            out_crop = network(
                trans=trans.to(network.device),
                dir=dir.to(network.device),
                ret=ret.to(network.device)
            )

        out = f.array(T.center_crop(out_crop, out_shape) * 255.).astype(dtype)
        if rotate180:
            out = np.rot90(out, -2, axes=(-3, -2))
        if channels == 1:
            out = out[..., 0]
        return {'x': batch['x'], 'y': batch['y'], 'out': out}

    def transfer(batch, network):
        b = get_outputs(batch, network)
        for x, y, out in zip(b['x'], b['y'], b['out']):
            height = min(out_section.shape[0] - y, out.shape[0])
            width = min(out_section.shape[1] - x, out.shape[1])
            try:
                out_section[y:y + height, x:x + width] = out[:height, :width]
            except:
                raise Exception(f"ERROR creating image at x={x}, y={y}, shape={out.shape}, height={height}, width={width}")

    print("Start mask generation...")
    for batch in tqdm(section_loader, desc=f"Rank {rank}"):
        transfer(batch, gen_model)

    return out_section


def save_section(
        out_section: np.ndarray,
        out_file: str,
        chunk_size: int = 256,
        pyramid_levels: int = 11,
        spacing: Tuple[float, float] = (1.0, 1.0),
        origin: Tuple[float, float] = (0.0, 0.0),
        modality: str = 'Cresyl',
        minimum: float = 0.0,
        maximum: float = 255.,
):
    print("Save section...")
    section = Section(image=out_section)
    section.spacing = spacing
    section.origin = origin
    section.modality = modality
    section.minimum = minimum
    section.maximum = maximum

    section.to_hdf5(out_file, chunk_size=chunk_size, num_levels=pyramid_levels, downscaling_factor=0.5)
    print(f"New section created at {out_file}")
