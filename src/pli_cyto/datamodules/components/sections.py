from typing import Optional, Tuple, List, Callable, Dict, Any, Union
import os
from collections import namedtuple

import math
import random

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import numpy as np

from atlaslib.io import stage_split_hdf5
from atlaslib.files import require_directory
from plio.section import Section

from pli_cyto import utils

# Distributed
from atlasmpi import MPI

comm = MPI.COMM_WORLD

log = utils.get_logger(__name__)


Coord = namedtuple("Coord", ('x', 'y'))


class SectionDataset(torch.utils.data.Dataset):

    def __init__(self, trans_file, dir_file, ret_file, patch_shape, out_shape, ram=True, norm_trans=None):
        # Expands the dataset to size input by repeating the provided ROIs
        # rois is a list of dicts with entries 'mask', 'ntrans', 'ret' and 'dir'
        super().__init__()
        self.ram = ram
        self.trans_section_mod = Section(path=trans_file)
        self.dir_section_mod = Section(path=dir_file)
        self.ret_section_mod = Section(path=ret_file)
        if ram:
            print("Load sections to RAM...")
            self.trans_section = np.array(self.trans_section_mod.image)
            self.dir_section = np.array(self.dir_section_mod.image)
            self.ret_section = np.array(self.ret_section_mod.image)
            print("All sections loaded to RAM")
        else:
            print("Do not load sections to RAM")
            self.trans_section = self.trans_section_mod.image
            self.dir_section = self.dir_section_mod.image
            self.ret_section = self.ret_section_mod.image

        self.norm_trans = norm_trans
        if not self.norm_trans:
            self.norm_trans = self.trans_section_mod.norm_value
        if not self.norm_trans:
            print("No normalization value for transmittance found. Make sure you passed NTransmittance!")
        self.brain_id = self.trans_section_mod.brain_id
        self.section_id = self.trans_section_mod.id
        self.section_roi = self.trans_section_mod.roi

        assert (patch_shape[0] - out_shape[0]) % 2 == 0  # Border symmetric
        assert (patch_shape[1] - out_shape[1]) % 2 == 0  # Border symmetric
        self.patch_shape = patch_shape
        self.out_shape = out_shape
        self.border = ((patch_shape[0] - out_shape[0]) // 2, (patch_shape[1] - out_shape[1]) // 2)
        self.shape = self.trans_section.shape

        self.coords = [Coord(x=x, y=y) for x in np.arange(0, self.shape[1], out_shape[1]) for y in
                       np.arange(0, self.shape[0], out_shape[0])]

    def __getitem__(self, i):
        x = self.coords[i].x
        y = self.coords[i].y
        b_y = self.border[0]
        b_x = self.border[1]
        pad_y_0 = max(b_y - y, 0)
        pad_x_0 = max(b_x - x, 0)
        pad_y_1 = max(y + (self.patch_shape[0] - b_y) - self.shape[0], 0)
        pad_x_1 = max(x + (self.patch_shape[1] - b_x) - self.shape[1], 0)
        trans_crop = torch.tensor(
            self.trans_section[max(0, y - b_y):min(self.shape[0], y + self.patch_shape[0] - b_y),
            max(0, x - b_x):min(self.shape[1], x + self.patch_shape[1] - b_x)]
        )
        if self.norm_trans is not None:
            trans_crop /= self.norm_trans
        ret_crop = torch.tensor(
            self.ret_section[max(0, y - b_y):min(self.shape[0], y + self.patch_shape[0] - b_y),
            max(0, x - b_x):min(self.shape[1], x + self.patch_shape[1] - b_x)]
        )
        dir_crop = torch.tensor(np.radians(
            self.dir_section[max(0, y - b_y):min(self.shape[0], y + self.patch_shape[0] - b_y),
            max(0, x - b_x):min(self.shape[1], x + self.patch_shape[1] - b_x)]
        ))

        trans_crop = F.pad(trans_crop, (pad_x_0, pad_x_1, pad_y_0, pad_y_1), mode='constant', value=0.0)
        dir_crop = F.pad(dir_crop, (pad_x_0, pad_x_1, pad_y_0, pad_y_1), mode='constant', value=0.0)
        ret_crop = F.pad(ret_crop, (pad_x_0, pad_x_1, pad_y_0, pad_y_1), mode='constant', value=0.0)

        return {'x': x, 'y': y, 'trans': trans_crop[None], 'dir': dir_crop[None], 'ret': ret_crop[None]}

    def __len__(self):
        return len(self.coords)


class ModalityCollection(object):

    load_func = {
        'cyto': lambda x, s: x[..., None].astype(np.float32) / 255. if len(x.shape) == 2 else x.astype(np.float32) / 255.,
        'trans': lambda x, s: (x / s.norm_value).astype(np.float32) if s.norm_value else x.astype(np.float32),
        'dir': lambda x, s: np.radians(x, dtype=np.float32),
        'ret': lambda x, s: x.astype(np.float32)
    }

    brain_id: str
    section_id: int
    section_roi: str
    shape: Tuple[int, ...]

    def __init__(
            self,
            files: Dict[str, Any],
            ram=True,
            driver: str = None,
            **h5kwargs
    ):
        super().__init__()

        self.files = files
        self.sections = None
        self.ram = ram
        self.h5kwargs = {'mode': 'r', 'driver': driver, **h5kwargs}

        self.setup()

    def setup(self):
        self.sections = {}
        n_files = {}
        for k, f in self.files.items():
            shm_dir = f"/dev/shm/{os.getlogin()}/{k}"
            if self.ram:
                require_directory(shm_dir)
                n_files[k] = os.path.join(shm_dir, os.path.basename(f))
                log.info(f"Create {n_files[k]}")
                if self.h5kwargs['driver'] == 'split':
                    utils.require_copy(f + "-m.h5", n_files[k] + "-m.h5", follow_symlinks=True)
                    utils.require_copy(f + "-r.h5", n_files[k] + "-r.h5", follow_symlinks=True)
                else:
                    utils.require_copy(f, n_files[k], follow_symlinks=True)
            else:
                if self.h5kwargs['driver'] == 'split':
                    n_files[k] = stage_split_hdf5(f, stage_dir=shm_dir)
                else:
                    n_files[k] = f
        
        comm.Barrier()

        for k, f in n_files.items():
            log.info(f"Open {f}, {self.h5kwargs}")
            self.sections[k] = Section(path=f, **self.h5kwargs)

        # Get metadata from next available section
        section = next(iter(self.sections.values()))
        self.brain_id = section.brain_id
        self.section_id = section.id
        self.section_roi = section.roi
        self.shape = section.shape
        
        for s in self.sections.values():
            assert s.shape == section.shape

    def _crop(self, arr: Any, x: int, y: int, size=256, max_size=5000 * 5000):
        if type(size) is tuple and len(size) == 2:
            if size[0] * size[1] <= max_size:
                return arr[y:y + size[0], x:x + size[1]]
            else:
                raise Exception("Crop size was too big. Adjust self.MAX_CROP_SIZE for larger sizes.")
        elif type(size) is int:
            if size ** 2 <= max_size:
                return arr[y:y + size, x:x + size]
            else:
                raise Exception("Crop size was too big. Adjust self.MAX_CROP_SIZE for larger sizes.")
        else:
            raise Exception("Wrong shape format. Expected tuple of shape (h, w) or int")


    def _center_crop(self, arr: Any, row: int, col:int, size: Union[int, Tuple[int, int]]):
        if type(size) is int:
            size = (size, size)
        if type(size) is tuple and len(size) == 2:
            pos_row = row - size[0] // 2
            pos_col = col - size[1] // 2
            return arr[pos_row:(pos_row + size[0]), pos_col:(pos_col + size[1])]
        else:
            raise Exception(f"Wrong format for size. Found type {type(size)}. Expected type int or tuple(int, int)")

    def get_crops(self, x, y, crop_size):
        return dict(
            (k, self.load_func[k](self._crop(s.image, x, y, crop_size), s))
            for k, s in self.sections.items()
        )

    def get_center_crops(self, row, col, crop_size):
        return dict(
            (k, self.load_func[k](self._center_crop(s.image, row, col, crop_size), s))
            for k, s in self.sections.items()
        )


class SectionSampler(Dataset):

    def __init__(
            self,
            cyto_files: List[str],
            trans_files: List[str],
            dir_files: List[str],
            ret_files: List[str],
            cyto_patch_size: Tuple[int, int],
            pli_patch_size: Tuple[int, int],
            transform: Optional[Callable] = None,
            n_samples: int = 1024,
            seed: int = None,
            ram: bool = False,
            driver: str = None,
    ):
        super().__init__()

        self.pli_sections = []
        for t, d, r in zip(trans_files, dir_files, ret_files):
            self.pli_sections.append(ModalityCollection({'trans': t, 'dir': d, 'ret': r}, ram, driver=driver))
        self.cyto_sections = []
        for c in cyto_files:
            self.cyto_sections.append(ModalityCollection({'cyto': c}, ram, driver=driver))
        assert len(self.pli_sections) == len(self.cyto_sections)
        self.n_sections = len(self.pli_sections)

        self.cyto_patch_size = tuple(cyto_patch_size)
        self.pli_patch_size = tuple(pli_patch_size)
        self.transform = transform
        self.n_samples = n_samples
        self.seed = seed

    def __getitem__(self, ix):
        if self.seed is not None:
            # Save random state
            state = random.getstate()
            random.seed(utils.better_seed([self.seed, ix]))

        section_ix = random.randint(0, self.n_sections - 1)
        pli_section = self.pli_sections[section_ix]
        cyto_section = self.cyto_sections[section_ix]

        assert pli_section.shape[:2] == cyto_section.shape[:2]

        h, w = pli_section.shape[:2]
        max_size = (max(self.cyto_patch_size[0], self.pli_patch_size[0]),
                    max(self.cyto_patch_size[1], self.pli_patch_size[1]))
        row = random.randint(math.ceil(max_size[0] / 2), h - math.ceil(max_size[0] / 2) - 1)
        col = random.randint(math.ceil(max_size[1] / 2), w - math.ceil(max_size[1] / 2) - 1)

        # TODO: Limit crops to foreground

        pli_crops = pli_section.get_center_crops(row, col, self.pli_patch_size)
        cyto_crop = cyto_section.get_center_crops(row, col, self.cyto_patch_size)
        patch = self.transform(pli_dict={
            'trans': pli_crops['trans'],
            'dir': pli_crops['dir'],
            'ret': pli_crops['ret'],
        }, image=cyto_crop['cyto'])

        # Reset random state
        if self.seed is not None:
            random.setstate(state)

        return {**patch['pli_dict'], 'cyto': patch['image']}

    def __len__(self):
        return self.n_samples