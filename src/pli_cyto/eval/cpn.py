import celldetection as cd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

import tqdm
from torch.cuda.amp import GradScaler, autocast

from . import names


def compute_precision(true_positives, false_positives):
    tp = true_positives
    fp = false_positives
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0


def compute_recall(true_positives, false_negatives):
    tp = true_positives
    fn = false_negatives
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0


def compute_f1(precision, recall):
    pr = precision
    rc = recall
    try:
        return (2 * pr * rc) / (pr + rc)
    except ZeroDivisionError:
        return 0


def compute_ap(true_positives, false_negatives, false_positives):
    tp = true_positives
    fn = false_negatives
    fp = false_positives
    try:
        return tp / (tp + fn + fp)
    except ZeroDivisionError:
        return 0


def sum_scores_row(row, key):
    score_dict = {}

    for area, r in row.items():       
        score_dict['tp'] = score_dict.get('tp', 0) + r[key]['tp']
        score_dict['fp'] = score_dict.get('fp', 0) + r[key]['fp']
        score_dict['fn'] = score_dict.get('fn', 0) + r[key]['fn']

    precision = compute_precision(score_dict['tp'], score_dict['fp'])
    recall = compute_recall(score_dict['tp'], score_dict['fn'])
    f1 = compute_f1(precision, recall)
    ap = compute_ap(score_dict['tp'], score_dict['fn'], score_dict['fp'])

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': ap
    }


def get_contours_size(contours, resolution, shrinkage_2d):
    from shapely import Polygon

    sizes = np.array([Polygon(c).area for c in contours]) * (resolution ** 2) * shrinkage_2d

    return sizes



def clean_contours(contours, locations, fourier, min_size, max_size, mask=None):
    from shapely import Polygon

    if len(contours) == 0:
        return contours, locations, fourier

    index1 = np.array([min_size < Polygon(c).area < max_size for c in contours])
    if mask is None:
        index = index1
    else:
        index2 = np.array([mask[int(l[1]), int(l[0])] for l in locations])
        index = index1 & index2

    clean_contours = contours[index]  # np.array([c for c in contours if min_size < Polygon(c).area < max_size])
    clean_locations = locations[index]
    clean_fourier = fourier[index]

    return clean_contours, clean_locations, clean_fourier


def compare_contours(contours1, contours2, shape):
    true_cells = cd.data.contours2labels(contours1, shape)
    pred_cells = cd.data.contours2labels(contours2, shape)

    results = cd.data.LabelMatcherList()
    results.append(cd.data.LabelMatcher(pred_cells, true_cells))

    return results


def evaluate(model, data_loader, device, use_amp, desc='Eval', progress=True, timing=False):
    """Evaluate model and return results."""
    model.eval()
    tq = tqdm(data_loader, desc=desc) if progress else data_loader
    results = cd.data.LabelMatcherList()
    times = []
    for batch_idx, batch in enumerate(tq):
        batch: dict = cd.to_device(batch, device)
        with autocast(use_amp):

            outputs: dict = model(batch['inputs'])

        out = cd.asnumpy(outputs)
        inp = cd.asnumpy(batch)
        targets = inp['targets']
        for idx in range(len(targets)):
            target = cd.data.channels_first2channels_last(targets[idx])
            prediction = cd.data.contours2labels(out['contours'][idx], target.shape[:2])
            results.append(cd.data.LabelMatcher(prediction, target))

    return results


def iou_scores(results, iou_threshs=(.2, .3, .4, .5), verbose=True):
    if verbose:
        print('iou thresh\t\t f1')
    out_f1 = {}
    for results.iou_thresh in iou_threshs:
        out_f1[results.iou_thresh] = results.avg_f1
        if verbose:
            print(results.iou_thresh, '\t\t\t', np.round(out_f1[results.iou_thresh], 3))

    return out_f1


def avg_iou_score(results, iou_threshs=(.2, .3, .4, .5), verbose=True):
    f1_scores = iou_scores(results, iou_threshs, verbose)
    final_f1 = np.mean(f1_scores.values()).round(3)
    if verbose:
        print('\nAverage F1 score:', '\t', final_f1)
    return final_f1


def apply_cpn(image, model, chunk_size, ghost_points, scores_upper_bound=0):
    from typhon.algorithms import apply_to_chunks

    out_list = []

    def process_chunk(chunk, valid_pixel_slice: slice, indices):
        # print("slice:", valid_pixel_slice)
        # print("indices:", indices)

        chunk_torch = torch.tensor(chunk, dtype=torch.float32).permute(2, 0, 1)[None] / 255.
        with torch.no_grad():
            model.eval()
            if scores_upper_bound == 0:
                out = cd.asnumpy(model(chunk_torch.to('cuda')))
            else:
                chunk_torch = chunk_torch.to('cuda')
                from torchvision.transforms.functional import gaussian_blur
                chunk_blur = gaussian_blur(chunk_torch, 5)
                upper_bound = (torch.mean(chunk_blur, dim=-3, keepdim=True) < scores_upper_bound).float()
                upper_bound = torch.nn.functional.interpolate(upper_bound, scale_factor=0.25)

                cd.imshow(upper_bound.cpu())

                out = cd.asnumpy(model(chunk_torch.to('cuda'), scores_upper_bound=upper_bound))

        valid_rows = (valid_pixel_slice[0].start <= out['locations'][0][:, 1]) * (
                    out['locations'][0][:, 1] < valid_pixel_slice[0].stop)
        valid_cols = (valid_pixel_slice[1].start <= out['locations'][0][:, 0]) * (
                    out['locations'][0][:, 0] < valid_pixel_slice[1].stop)

        valid_contours = list(valid_rows * valid_cols)
        n_found = sum(valid_contours)

        if n_found > 0:
            out = dict((k, [np.array([x for x, m in zip(v[0], valid_contours) if m])]) for k, v in out.items())
            pos_row = max(0, indices[0] * chunk_size - ghost_points)
            pos_col = max(0, indices[1] * chunk_size - ghost_points)

            # print(out.keys())

            out['contours'][0] = out['contours'][0] + np.array([pos_col, pos_row])
            out['locations'][0] = out['locations'][0] + np.array([pos_col, pos_row])
        else:
            out = {}

        return out

    # Define function to apply to each chunk
    def _function(chunk, indices, valid_pixel_slice, **kwargs):
        out = process_chunk(chunk, valid_pixel_slice, indices)
        out_list.append(out)

    # Apply function to each chunk
    apply_to_chunks(
        image,
        function=_function,
        write_back=False,
        pass_valid_pixel_slice=True,
        chunk_size=chunk_size,
        ghost_points=ghost_points,
        input_dimension=2,
        rank=0,
        size=1,
    )

    valid_outputs = [s for s in out_list if len(s) > 0]
    if len(valid_outputs) > 0:
        contours = np.concatenate([s['contours'][0] for s in valid_outputs])
        locations = np.concatenate([s['locations'][0] for s in valid_outputs])
        fourier = np.concatenate([s['fourier'][0] for s in valid_outputs])
    else:
        contours = np.array([])
        locations = np.array([])
        fourier = np.array([])

    return contours, locations, fourier


def compute_cpn(
        prediction_path,
        gm_class = [2, 5],
        mask_path=f"datasets/vervet1818-stained/data/spline/mask/",
        margin=128,
        chunk_size=1024,
        ghost_points=128,
        median_blur_before_cpn=3,
        model_name='vacumu_CpnResNeXt101UNet-f33b2634bb51f299',
        score_thresh=0.9,
        min_size=0,
        max_size=99999999,
):
    import os
    from pli_cyto.eval.cpn import apply_cpn, clean_contours, compare_contours
    import celldetection as cd
    import cv2

    def load_image(path, slices):
        import h5py as h5
        with h5.File(path, 'r') as f:
            image = f['Image'][slices]
        return image

    model = cd.fetch_model(model_name)
    model = model.to('cuda')
    model.eval()
    model.score_thresh = score_thresh

    image = load_image(
        prediction_path,
        (slice(margin, -margin), slice(margin, -margin)),
    )

    mask = load_image(
        mask_path,
        (slice(margin, -margin), slice(margin, -margin)),
    )
    gm_mask = np.isin(mask, gm_class)

    contours, locations, fourier = apply_cpn(
        cv2.medianBlur(image, ksize=median_blur_before_cpn),
        model,
        chunk_size,
        ghost_points
    )

    clean_contours, clean_locations, clean_fourier = clean_contours(
        contours, locations, fourier,
        min_size, max_size, gm_mask,
    )

    return clean_contours, clean_locations, clean_fourier

def eval_cpn(
        prediction_path,
        indices,
        gm_class = [2, 5],
        test_folder="datasets/vervet1818-stained/data/spline/",
        mask_path=f"datasets/vervet1818-stained/data/spline/mask/",
        json_path="rois.json",
        margin=128,
        chunk_size=1024,
        ghost_points=128,
        median_blur_before_cpn=3,
        model_name='vacumu_CpnResNeXt101UNet-f33b2634bb51f299',
        score_thresh=0.9,
        iou_thresh=0.5,
        bin_sizes=[0, 100, 500, 99999],
        verbose=True
):
    import pandas as pd
    import os
    from pli_cyto.eval.cpn import apply_cpn, clean_contours, compare_contours
    import celldetection as cd
    import cv2

    def load_image(path, slices):
        import h5py as h5
        with h5.File(path, 'r') as f:
            image = f['Image'][slices]
        return image

    model = cd.fetch_model(model_name)
    model = model.to('cuda')
    model.eval()
    model.score_thresh = score_thresh

    roi_frame = pd.read_json(os.path.join(test_folder, json_path), orient='index')
    if verbose:
        print(roi_frame.head())

    metrics = {}
    for index in indices:
        cyto_fake = load_image(
            os.path.join(prediction_path, f"{index:04d}_CresylStyle.h5"),
            (slice(margin, -margin), slice(margin, -margin)),
        )

        row = roi_frame.loc[index]
        cyto_real = load_image(
            os.path.join(test_folder, row.cyto_spline),
            (slice(margin, -margin), slice(margin, -margin)),
        )

        mask = load_image(
            os.path.join(mask_path, f"{index:04d}_layers.h5"),
            (slice(margin, -margin), slice(margin, -margin)),
        )
        gm_mask = np.isin(mask, gm_class)

        if verbose:
            print(f"Apply CPN to {row.cyto_spline}")
        contours_real, locations_real, fourier_real = apply_cpn(
            cv2.medianBlur(cyto_real, ksize=median_blur_before_cpn),
            model,
            chunk_size,
            ghost_points
        )
        contours_fake, locations_fake, fourier_fake = apply_cpn(
            cv2.medianBlur(cyto_fake, ksize=median_blur_before_cpn),
            model,
            chunk_size,
            ghost_points
        )

        metrics[names[index]] = {}

        for i in range(len(bin_sizes) - 1):
            min_size = bin_sizes[i]
            max_size = bin_sizes[i + 1]

            if verbose:
                print(f"Evaluate range from {min_size} to {max_size}")

            print("1")
            contours_real_bin, _, _ = clean_contours(
                contours_real, locations_real, fourier_real,
                min_size, max_size, gm_mask,
            )
            contours_fake_bin, _, _ = clean_contours(
                contours_fake, locations_fake, fourier_fake,
                min_size, max_size, gm_mask,
            )

            if verbose:
                print("Print real detection...")
                cd.vis.show_detection(
                    cyto_real,
                    contours=contours_real_bin,
                    figsize=(8, 8),
                    contour_linestyle='-',
                    cmap='gray' if cyto_real.shape[2] == 1 else ...,
                    contour_kwargs={
                        'color': 'black',
                    }
                )
                plt.show()

                print("Print fake detection...")
                cd.vis.show_detection(
                    cyto_fake,
                    contours=contours_fake_bin,
                    figsize=(8, 8),
                    contour_linestyle='-',
                    cmap='gray' if cyto_fake.shape[2] == 1 else ...,
                    contour_kwargs={
                        'color': 'black',
                    }
                )
                plt.show()

            print("2")
            real_bin_vs_fake = compare_contours(contours_real, contours_fake_bin, cyto_real.shape[:2])
            real_bin_vs_fake.iou_thresh = iou_thresh

            true_pos = real_bin_vs_fake[0].true_positives
            false_pos = real_bin_vs_fake[0].false_positives

            print("TP", true_pos, "FP", false_pos)

            print("3")
            real_vs_fake_bin = compare_contours(contours_real_bin, contours_fake, cyto_real.shape[:2])
            real_vs_fake_bin.iou_thresh = iou_thresh

            false_neg = real_vs_fake_bin[0].false_negatives

            print("FN", false_neg)

            precision = compute_precision(true_pos, false_pos)
            recall = compute_recall(true_pos, false_neg)
            f1 = compute_f1(precision, recall)
            ap = compute_ap(true_pos, false_neg, false_pos)

            metric_dict = {
                'tp': true_pos,
                'fp': false_pos,
                'fn': false_neg,
                'f1': f1,
                'ap': ap,
                'recall': recall,
                'precision': precision,
            }

            print(metric_dict)

            metrics[names[index]][(min_size, max_size)] = metric_dict

    return metrics
