from typing import Callable, List, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import pli_cyto.utils.functions as f


def adversarial_ce_loss(disc: nn.Module, image: Tensor, target: bool = True):
    valid = torch.full((len(image), 1), target, dtype=image.dtype, device=image.device)
    loss = F.binary_cross_entropy(disc(image), valid)

    return loss


def adversarial_wasserstein_loss(disc: nn.Module, image: Tensor, target: bool = True):
    prefix = (1. - 2. * target)
    loss = prefix * torch.mean(disc(image))

    return loss


# Adapted from Cornelius and Eric
def style_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        style_net: nn.Module,
        ft_weights: List[float],
        loss_fn: Callable[..., torch.Tensor],
        target_mask: torch.Tensor = None
):
    if target_mask is not None:
        target[~target_mask] = torch.nan

    style_net.eval()
    ft_pred = style_net(pred)
    with torch.no_grad():
        ft_target = style_net(target)

    s_loss = 0.
    for w, ft_p, ft_t in zip(ft_weights, ft_pred, ft_target):
        if target_mask is not None:
            mask = ~torch.isnan(ft_t)
            ft_t[~mask] = 0.0
        else:
            mask = None
        g_p = f.gram_matrix(ft_p, mask)  # (N, C, C)
        g_t = f.gram_matrix(ft_t, mask)  # (N, C, C)
        s_loss += w * loss_fn(g_p, g_t)

    return s_loss


# Reconstruction losses
#######################

def mse_registration_loss(target: Tensor, prediction: Tensor, grid: Tensor, min_overlap: int, loss: Callable,
                          keep_best: int = -1, register_target=False):
    if register_target:
        best_e, best_target, best_mask = f.mse_dft_grid(prediction.detach(), target, grid, circular=False,
                                                        min_overlap=min_overlap)
        best_prediction = prediction

    else:
        best_e, best_prediction, best_mask = f.mse_dft_grid(target, prediction, grid, circular=False,
                                                            min_overlap=min_overlap)
        best_target = target

    if 0 <= keep_best < len(best_e):
        best_ix = torch.argsort(best_e)[:keep_best]
        return loss(best_target[best_ix], best_prediction[best_ix], best_mask[best_ix])
    else:
        return loss(best_target, best_prediction, best_mask)


def circ_mse_loss(target: Tensor, prediction: Tensor, keep_best: int = -1):
    circ_mse = torch.mean(f.circ_mse_dft(target, prediction, correct=False, circular=False), dim=1)  # N, C, H, W
    best_mse = torch.min(circ_mse.view(circ_mse.shape[0], -1), dim=-1)

    if keep_best == -1:
        return torch.mean(best_mse)
    else:
        best_mse = torch.sort(torch.tensor(best_mse), descending=False)[0]
        return torch.mean(best_mse[:keep_best])


def mae_loss(image: Tensor, prediction: Tensor, mask: Tensor = None):
    if mask is None:
        loss = torch.mean(torch.abs(image - prediction))
    else:
        loss = torch.mean(torch.abs(image[mask] - prediction[mask]))

    return loss


def mse_loss(image: Tensor, prediction: Tensor, mask: Tensor = None):
    if mask is None:
        loss = torch.mean((image - prediction) ** 2)
    else:
        loss = torch.mean((image[mask] - prediction[mask]) ** 2)

    return loss


# Adversarial Loss #
####################


def adversarial_ws(
        disc: Any,
        batch: torch.Tensor,
        target: bool,
):
    logits = disc.forward(batch)

    # Define targets for cross entropy loss
    if target:
        l = -torch.mean(logits)
    else:
        l = torch.mean(logits)

    return l


def adversarial_ce(
        disc: Any,
        batch: torch.Tensor,
        target: bool,
):
    logits = disc.forward(batch)

    # Define targets for cross entropy loss
    if target:
        targets = torch.ones_like(logits)
        targets = targets.type_as(logits)
    else:
        targets = torch.zeros_like(logits)
        targets = targets.type_as(logits)

    l = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    l = torch.mean(l)

    return l