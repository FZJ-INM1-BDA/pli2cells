from typing import List, Tuple, Iterable
import math

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as T
import numpy as np


def array(torch_tensor, permute=True):
    """
    Converts a torch.Tensor to a numpy array on the CPU and permutes the channels
    in a numpy friendly format.

    Parameters:
    -----------
    :param torch_tensor: torch.Tensor
        Input torch tensor.

    :param permute: boolean
        If to permute the dimensions to the NumPy convention (N x H x W x C).
        This paramter has only an effect if the number of dimensions is 3 or 4.

    Returns
    -------
    :return:
        np_array: np.ndarray
            Output NumPy array
    """
    assert (type(torch_tensor) is torch.Tensor)
    if len(torch_tensor.shape) == 4 and permute:
        np_array = torch_tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    elif len(torch_tensor.shape) == 3 and permute:
        np_array = torch_tensor.detach().cpu().permute(1, 2, 0).numpy()
    else:
        np_array = torch_tensor.detach().cpu().numpy()
    return np_array


def tensor(np_array, device='cpu', dtype=None, permute=True):
    """
    Converts a numpy.ndarray to a torch.tensor on the given device. Optional
    a datatype can be defined to convert the input to.

    Parameters:
    -----------
    :param np_array: np.ndarray
        Input NumPy array

    :param device: str
        Name of the device to transfer the tensor to. Default is 'cpu'

    :param dtype: torch.dtype (optional)
        The datatype to transform the NumPy array data to.

    :param permute: boolean
        If to permute the dimensions to the PyTorch convention (N x C x H x W).
        This paramter has only an effect if the number of dimensions is 3 or 4.

    Returns
    -------
    :return:
        torch_tensor: torch.Tensor
            The converted output torch Tensor
    """
    assert (type(np_array) is np.ndarray)

    if len(np_array.shape) == 4 and permute:
        np_tensor = np_array.transpose(0, 3, 1, 2)
    elif len(np_array.shape) == 3 and permute:
        np_tensor = np_array.transpose(2, 0, 1)
    else:
        np_tensor = np_array

    if dtype:
        torch_tensor = torch.tensor(np_tensor, device=device, dtype=dtype)
    else:
        torch_tensor = torch.tensor(np_tensor, device=device)

    return torch_tensor


def unravel_index(index, shape):
    index = int(index)
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(out[::-1])


def get_index(measure: torch.Tensor, mode='max'):
    if mode == 'min':
        return unravel_index(torch.argmin(measure), measure.shape)
    if mode == 'max':
        return unravel_index(torch.argmax(measure), measure.shape)


def get_rotation_grid(
        angles: Iterable,
        shape: Tuple[int, int],
        channels: int = 1
):
    thetas = []
    for a in angles:
        phi = -torch.deg2rad(torch.tensor(a))
        thetas.append(torch.tensor([
            [torch.cos(phi), torch.sin(phi), 0],
            [-torch.sin(phi), torch.cos(phi), 0]
        ]))
    thetas = torch.stack(thetas)

    out_size = [len(thetas), channels, shape[0], shape[1]]

    return F.affine_grid(thetas, out_size, align_corners=False)


def gram_matrix(features: torch.Tensor, mask: torch.Tensor = None):
    """

    Calculation of gram matrix for features. This function can be called with a mask to compute the matrix only for masked features.

    :param features: Features of shape (N, C, H, W)
    :param mask: (Optional) Mask of same shape as features (N, C, H, W). All pixels with value False will be ignored

    :return: Gram matrix of shape (N, C, C)
    """
    _, c, h, w = features.shape

    if mask is not None:
        features = features * mask
        counts = torch.sum(mask, dim=(-3, -2, -1))[:, None, None] + 1e-6
    else:
        counts = c * h * w

    ft = torch.flatten(features, -2, -1)  # N, C, HW
    ft_T = ft.transpose(-2, -1)  # N, HW, C

    g = (ft @ ft_T) / counts  # N, C, C

    return g


def circ_cross_corr(x, y):
    from torch.fft import rfft2, irfft2

    x_dft = rfft2(x)
    y_dft = rfft2(y)

    hadamard = x_dft * torch.conj(y_dft)
    ccc = irfft2(hadamard, s=x.shape[-2:])

    return ccc


def circ_mse(x, y, circular=True):
    """
    Calculates the MSE between x and y for every possible (circular) translation of y.

    The size of y needs to be smaller or equal to x, i.e. h <= H and w <= W.

    If circulation of translation is not wanted make sure that h <= H / 2, w <= W / 2!

    Uses an exhaustive implementation via DFT of complexity O(H**2 * W**2)

    :param x: torch.Tensor of shape (..., H, W)
    :param y: torch.Tensor of shape (..., h, w)
    :param circular: If performing circular MSE or not
    :return: MSE error for every possible (circular) translation
    """

    h, w = y.shape[-2:]

    start_row = 0 if circular else 1 - h
    start_col = 0 if circular else 1 - w

    e = torch.empty((x.shape[-2] - start_row, x.shape[-1] - start_col)) * 100
    for r in np.arange(start_row, x.shape[-2], dtype=int):
        for c in np.arange(start_col, x.shape[-1], dtype=int):
            if circular:
                x_roll = torch.roll(x, (-r, -c), dims=(-2, -1))
                e[..., r, c] = torch.mean((x_roll[..., :h, :w] - y) ** 2, dim=(-2, -1))
            else:
                x_top = max(0, r)
                x_left = max(0, c)
                x_bot = min(r + h, x.shape[-2])
                x_right = min(c + w, x.shape[-1])
                x_crop = x[..., x_top:x_bot, x_left:x_right]
                if r <= 0 and c <= 0:
                    m = torch.sum((x_crop - y[..., -x_crop.shape[-2]:, -x_crop.shape[-1]:]) ** 2, dim=(-2, -1)) / \
                        (x_crop.shape[-1] * x_crop.shape[-2])
                elif r >= 0 and c <= 0:
                    m = torch.sum((x_crop - y[..., :x_crop.shape[-2], -x_crop.shape[-1]:]) ** 2, dim=(-2, -1)) / \
                        (x_crop.shape[-1] * x_crop.shape[-2])
                elif r <= 0 and c >= 0:
                    m = torch.sum((x_crop - y[..., -x_crop.shape[-2]:, :x_crop.shape[-1]]) ** 2, dim=(-2, -1)) / \
                        (x_crop.shape[-1] * x_crop.shape[-2])
                else:
                    m = torch.sum((x_crop - y[..., :x_crop.shape[-2], :x_crop.shape[-1]]) ** 2, dim=(-2, -1)) / \
                        (x_crop.shape[-1] * x_crop.shape[-2])
                e[..., r, c] = m

    return e


def circ_mse_dft(x, y, correct=True, circular=True):
    """
    Calculates the MSE between x and y for every possible (circular) translation of y.

    The size of y needs to be smaller or equal to x, i.e. h <= H and w <= W.
    If circulation of translation is not wanted set circular=False.

    Uses an efficient implementation via DFT of complexity O(H * W * (log(H) + log(W)).
    To dot that it uses an expansion of (x - y) ** 2 as x ** 2 - 2 * x * y + y ** 2 and performs
    the correlation part x * y by the Fourier Transform Cross Correlation Theorem.

    :param x: torch.Tensor of shape (N, C, H, W)
    :param y: torch.Tensor of shape (N, C, h, w), it has to fulfill h <= H and w <= W.
    :param correct: If to correct MSE for unused values of x ** 2 and y ** 2
    :param circular: If performing circular MSE or not
    :return: MSE error for every possible (circular) translation of shape (N, C, H, W) or (N, C, H + p, W + p) if not circular
    """

    if correct:
        # If correction is performed later, ignore every padded pixel
        fill_value = 0.
    else:
        # If no correction is performed later get a best guess for padded pixels by mean of x
        # However, this can lead to smoothing of pixels that do not overlap with x
        fill_value = float(torch.mean(x))

    if circular:
        # Dont pad x
        pad_x_col = 0
        pad_x_row = 0
        x_pad = x
    else:
        # Pad x by shape of y - 1, so filtering becomes non-circular
        pad_x_col = y.shape[-1] - 1
        pad_x_row = y.shape[-2] - 1
        x_pad = F.pad(x, (0, pad_x_col, 0, pad_x_row), value=fill_value)

    pad_y_col = x_pad.shape[-1] - y.shape[-1]
    pad_y_row = x_pad.shape[-2] - y.shape[-2]
    y_pad = F.pad(y, (0, pad_y_col, 0, pad_y_row), value=fill_value)

    xy = circ_cross_corr(x_pad, y_pad)
    x_sq = torch.sum(x_pad ** 2, dim=(-1, -2), keepdim=True)
    y_sq = torch.sum(y_pad ** 2, dim=(-1, -2), keepdim=True)

    x_corr = 0.0
    y_corr = 0.0
    n = y.shape[-1] * y.shape[-2]

    if correct:
        if not circular:
            # Sum values of y ** 2 that are not aligned with values of x
            mask_pad_x = F.pad(torch.zeros(x.shape[-2:], device=y_pad.device), (0, pad_x_col, 0, pad_x_row), value=1.)
            y_corr = circ_cross_corr(mask_pad_x, y_pad ** 2)

            # Get intersected pixels that are not padded by either x or y
            mask_pad_y_center = F.pad(torch.ones(y.shape[-2:], device=mask_pad_x.device), (0, pad_y_col, 0, pad_y_row),
                                      value=0.)
            n = circ_cross_corr(1. - mask_pad_x, mask_pad_y_center)

        if pad_y_row > 0. or pad_y_col > 0.:
            # Sum values of x ** 2 that are not aligned with values of y
            mask_pad_y = F.pad(torch.zeros(y.shape[-2:], device=x_pad.device), (0, pad_y_col, 0, pad_y_row), value=1.)
            x_corr = circ_cross_corr(x_pad ** 2, mask_pad_y)

    e = (x_sq + -2 * xy + y_sq - x_corr - y_corr) / n

    return e


def mse_dft_grid(
        x: torch.Tensor,
        y: torch.Tensor,
        grid: torch.Tensor,
        circular: bool = True,
        min_overlap: int = 0,
        return_error_map: bool = False,
        parallel: bool = False,
        return_indexes: bool = False,
):
    """
    Calculates the MSE between x and y for every possible (circular) translation of y and every transform specified
    as a 2D sampling grid (see https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
    If circulation of translation is not wanted, set circular=False.

    The size of y should be smaller or equal to x, i.e. h <= H and w <= W.

    Uses an efficient implementation via DFT of complexity O(H * W * (log(H) + log(W)).
    To dot that it uses an expansion of (x - y) ** 2 as x ** 2 - 2 * x * y + y ** 2 and performs
    the correlation part x * y by the Fourier Transform Cross Correlation Theorem.

    :param x: torch.Tensor of shape (N, C, H, W)
    :param y: torch.Tensor of shape (N, C, h, w), it should fulfill h <= H and w <= W.
    :param grid: Sampling grid as a tensor of indices of shape (T, H, W, 2), where T is the number of transforms
    :param circular: If performing circular MSE or not
    :param min_overlap: If circular is False, only regards MSE computation for a minimum overlap of pixels between
    target x and prediction y. Sets all other values to the maximum MSE.
    :param return_error_map: If True also return the complete error map of shape (N, T, C, H, W)
    :param parallel: If True executes the grid_sample as batch,Warning! this can lead to high memory consumption.
    :param return_indexes: If True also return the angle index and the shift coordinates to obtain the minimum MSE.

    :return: MSE error for every possible (circular) translation and T transformations of shape (N, T, H, W)
    or (N, T, H + p, W + p) if not circular with some padding p
    """
    import torch.nn.functional as F

    x_pad, y_pad, mask_y_pad_stack, mask_x_pad, params = pad_inputs(x, y, grid, circular)
    pad_y_col = params['pad_y_col']
    pad_y_row = params['pad_y_row']

    # Get intersected pixels that are not padded by either x or y
    if circular:
        n = y.shape[-1] * y.shape[-2]
    else:
        n = torch.clip(circ_cross_corr(mask_x_pad, mask_y_pad_stack), 1.)  # T, C, H, W

    out_e = []
    best_e = []
    best_y = []
    best_mask = []
    all_best_ix = []

    if parallel:  # parallel
        # CONVERT INPUT FROM B, C, H, W -> BxC, H, W
        y_shape = y_pad.shape
        y_sample_all = torch.flatten(y_pad, 0, 1)
        y_pad_stack = F.grid_sample(
            y_sample_all[None].expand(grid.shape[0], *y_sample_all.shape),
            grid,
            mode='nearest',
            align_corners=False
        )
        y_pad_stack = F.pad(
            y_pad_stack,
            (0, pad_y_col, 0, pad_y_row),
            value=0.
        )

        y_pad_stack = torch.nn.Unflatten(1, (y_shape[0], y_shape[1]))(y_pad_stack)
        y_pad_stack = torch.permute(y_pad_stack, dims=(1, 0, 2, 3, 4))  # B,T,C,H,W

        x_pad = x_pad.unsqueeze(1)
        agg = torch.sum(x_pad ** 2, dim=(-1, -2), keepdim=True)  # x_sq
        agg = agg + torch.sum(y_pad_stack ** 2, dim=(-1, -2), keepdim=True)  # y_sq
        agg = agg - 2 * circ_cross_corr(x_pad, y_pad_stack)  # xy

        if y_pad_stack.shape != y.shape:
            agg = agg - circ_cross_corr((1. - mask_x_pad[None, None]), y_pad_stack ** 2)

        if x_pad.shape != x.shape:
            agg = agg - circ_cross_corr(x_pad ** 2, (1 - mask_y_pad_stack))

        e = agg / n
        e = torch.mean(e, dim=-3)  # B x T x H x W

        if min_overlap > 0 and not circular:
            n_sum = torch.mean(n, dim=-3)

            n_sum = n_sum.expand(e.shape[0], *n_sum.shape)
            e[n_sum < min_overlap] = torch.max(e[n_sum >= min_overlap])

        best_ix = e.view(e.shape[0], -1).argmin(1)
        best_ix_list = [[n_x, *unravel_index(best_ix_, e.shape[1:])] for n_x, best_ix_ in enumerate(best_ix)]
        best_ix = torch.tensor(best_ix_list)
        all_best_ix = best_ix[:, 1:]

        r_rows = best_ix[:, -1]
        r_cols = best_ix[:, -2]

        best_y_ = [
            torch.roll(y_pad_stack[best_ix_[0], best_ix_[1]], (r_row, r_col), dims=(-1, -2))[:, :x.shape[-2],
            :x.shape[-1]] for best_ix_, r_row, r_col in zip(best_ix, r_rows, r_cols)
        ]

        best_mask_ = [
            torch.roll(mask_y_pad_stack[best_ix_[1]], (r_row, r_col), dims=(-1, -2))[:, :x.shape[-2], :x.shape[-1]] for
            best_ix_, r_row, r_col in zip(best_ix, r_rows, r_cols)
        ]

        best_e = torch.tensor([e[tuple(best_ix_)] for best_ix_ in best_ix_list]).to(x)

        best_y = torch.stack(best_y_)
        best_mask = torch.stack(best_mask_) > 0.5

        if return_error_map:
            out_e = torch.roll(
                e,
                (int((y_pad.shape[-1] - y.shape[-1]) / 2.), int((y_pad.shape[-2] - y.shape[-2]) / 2.)),
                dims=(-2, -1)
            )

    else:
        for ix, y_sample in enumerate(y_pad):  # (C, H, W)

            y_pad_stack = F.grid_sample(
                y_sample[None].expand(grid.shape[0], *y_sample.shape),
                grid, mode='nearest',
                align_corners=False
            )  # T, C, H, W
            y_pad_stack = F.pad(
                y_pad_stack,
                (0, pad_y_col, 0, pad_y_row),
                value=0.
            )

            xy = circ_cross_corr(x_pad[ix], y_pad_stack)
            x_sq = torch.sum(x_pad[ix] ** 2, dim=(-1, -2))[..., None, None]
            y_sq = torch.sum(y_pad_stack ** 2, dim=(-1, -2))[..., None, None]

            x_corr = 0.0
            y_corr = 0.0
            if y_pad_stack.shape != y.shape:
                # Sum values of y ** 2 that are not aligned with values of x
                y_corr = circ_cross_corr((1. - mask_x_pad), y_pad_stack ** 2)

            if x_pad.shape != x.shape:
                # Sum values of x ** 2 that are not aligned with values of y
                x_corr = circ_cross_corr(x_pad[ix] ** 2, (1 - mask_y_pad_stack))

            e = (x_sq + -2 * xy + y_sq - x_corr - y_corr) / n  # T, C, H, W
            e = torch.mean(e, dim=-3)  # T x H x W

            if min_overlap > 0 and not circular:
                n_sum = torch.mean(n, dim=-3)
                e[n_sum < min_overlap] = torch.max(e[n_sum >= min_overlap])

            best_ix = get_index(e, 'min')
            all_best_ix.append(best_ix)

            r_row = best_ix[-1]
            r_col = best_ix[-2]

            best_y.append(
                torch.roll(y_pad_stack[best_ix[0]], (r_row, r_col), dims=(-1, -2))[:, :x.shape[-2], :x.shape[-1]]
            )
            best_mask.append(
                torch.roll(mask_y_pad_stack[best_ix[0]], (r_row, r_col), dims=(-1, -2))[:, :x.shape[-2], :x.shape[-1]]
            )
            best_e.append(e[best_ix])

            if return_error_map:
                out_e.append(
                    torch.roll(
                        e,
                        (int((y_pad.shape[-1] - y.shape[-1]) / 2.), int((y_pad.shape[-2] - y.shape[-2]) / 2.)),
                        dims=(-2, -1)
                    )
                )

        best_e = torch.stack(best_e)
        best_y = torch.stack(best_y)
        best_mask = torch.stack(best_mask) > 0.5
        all_best_ix = torch.tensor(all_best_ix)

        if return_error_map:
            out_e = torch.stack(out_e)

    output = best_e, best_y, best_mask
    if return_error_map:
        output += (out_e,)

    if return_indexes:
        output += (all_best_ix,)

    return output


def pad_inputs(x, y, grid, circular):
    import torch.nn.functional as F

    if circular:
        # Make x_pad the same size as y_pad
        pad_x_row = max(0, grid.shape[-3] - x.shape[-2])
        pad_x_col = max(0, grid.shape[-2] - x.shape[-1])
    else:
        # Pad x by shape of y - 1, so filtering becomes non-circular
        pad_x_row = grid.shape[-3] - 1
        pad_x_col = grid.shape[-2] - 1
    x_pad = F.pad(
        x,
        (0, pad_x_col, 0, pad_x_row),
        value=0.
    )  # N, C, H, W

    # Pad y in every direction if not aligned with shape of grid
    pad_y_row_center = max(0, (grid.shape[-3] - y.shape[-2]) / 2.)
    pad_y_col_center = max(0, (grid.shape[-2] - y.shape[-1]) / 2.)
    pad_y_row = x_pad.shape[-2] - grid.shape[-3]
    pad_y_col = x_pad.shape[-1] - grid.shape[-2]
    y_pad = F.pad(
        y,
        (math.floor(pad_y_col_center), math.ceil(pad_y_col_center), math.floor(pad_y_row_center),
         math.ceil(pad_y_row_center)),
        value=0.
    )  # N, C, H, W
    mask_y_pad = F.pad(
        torch.ones(y.shape[-3:], device=y.device),
        (math.floor(pad_y_col_center), math.ceil(pad_y_col_center), math.floor(pad_y_row_center),
         math.ceil(pad_y_row_center)),
        value=0.
    )  # C, H, W
    mask_y_pad_stack = F.grid_sample(
        mask_y_pad[None].expand(grid.shape[0], *mask_y_pad.shape),
        grid,
        mode='nearest',
        align_corners=False
    )  # T, C, H, W
    mask_y_pad_stack = F.pad(
        mask_y_pad_stack,
        (0, pad_y_col, 0, pad_y_row),
        value=0.
    )  # T, C, H, W

    mask_x_pad = F.pad(
        torch.ones(x.shape[-3:], device=x_pad.device),
        (0, pad_x_col, 0, pad_x_row),
        value=0.
    )  # C, H, W

    params = dict(pad_x_col=pad_x_col, pad_x_row=pad_x_row, pad_y_col=pad_y_col, pad_y_row=pad_y_row)

    return x_pad, y_pad, mask_y_pad_stack, mask_x_pad, params


def phase_corr(x, y, circular=False):
    """
    Calculates the phase correlation between x and y for every possible circular translation of y.

    The size of y needs to be smaller or equal to x, i.e. h <= H and w <= W

    :param x: torch.Tensor of shape (..., H, W)
    :param y: torch.Tensor of shape (..., h, w)
    :param circular: If to apply circular correlation or not. Zero pad signal if not.
    :return:
    """

    from torch.fft import rfft2, irfft2
    import torch.nn.functional as F

    pad_x = (x.shape[-1] - y.shape[-1]) / 2.
    pad_y = (x.shape[-2] - y.shape[-2]) / 2.
    x_pad = x
    y_pad = F.pad(y, (math.floor(pad_x), math.ceil(pad_x),
                      math.floor(pad_y), math.ceil(pad_y)), value=torch.mean(y))

    if not circular:
        x_pad = F.pad(x_pad, (0, x.shape[-1], 0, x.shape[-2]))
        y_pad = F.pad(y_pad, (0, y.shape[-1], 0, y.shape[-2]))

    x_dft = rfft2(x_pad)
    y_dft = rfft2(y_pad)

    hadamard = x_dft * torch.conj(y_dft)

    pc = irfft2(hadamard / hadamard.abs())
    pc = torch.roll(pc, (math.floor(pad_y), math.floor(pad_x)), dims=(-2, -1))

    return pc
