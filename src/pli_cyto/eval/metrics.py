import numpy as np
import pandas as pd

from torchvision.transforms import ToTensor

from . import names


def mse(image1, image2, mask=None):
    from skimage.metrics import mean_squared_error

    if mask is not None:
        mse = mean_squared_error(image1[mask], image2[mask])
    else:
        mse = mean_squared_error(image1, image2)

    return mse


def rmse(image1, image2, mask=None):
    return np.sqrt(mse(image1, image2, mask))


def ssim(image1, image2, mask=None):
    from skimage.metrics import structural_similarity

    image1_masked = image1.copy()
    image2_masked = image2.copy()
    
    if mask is not None:
        image1_masked[~mask] = 0
        image2_masked[~mask] = 0

    _, S = structural_similarity(image1_masked, image2_masked,
                                 multichannel=True, full=True, channel_axis=-1)

    if mask is not None:
        ssim_result = np.mean(S[mask])
    else:
        ssim_result = np.mean(S)

    return ssim_result


def mi(image1, image2, mask=None):
    from sklearn.metrics import mutual_info_score
    import cv2

    # Convert to gray
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    if mask is not None:
        return mutual_info_score(image1[mask].flatten(), image2[mask].flatten())
    else:
        return mutual_info_score(image1.flatten(), image2.flatten())


def eval_pixels(
        prediction_path,
        indices,
        gm_class=[2],
        test_folder="datasets/vervet1818-stained/data/spline/",
        json_path="rois.json",
        mask_path=f"datasets/vervet1818-stained/data/spline/mask/",
        margin=128,
        verbose=True,
        gray_scale=False,
):
    from tqdm import tqdm
    import os
    from plio.section import Section

    roi_frame = pd.read_json(os.path.join(test_folder, json_path), orient='index')

    metrics = {}
    for index in indices:
        mask_file = os.path.join(mask_path, f"{index:04d}_layers.h5")
        mask_section = Section(path=mask_file)
        mask_crop = mask_section.image[margin:-margin, margin:-margin]

        tissue_mask = np.isin(mask_crop, gm_class)

        cyto_fake_section = Section(os.path.join(prediction_path, f"{index:04d}_CresylStyle.h5"))
        cyto_fake = cyto_fake_section.image[margin:-margin, margin:-margin]

        if gray_scale:
            cyto_fake = np.mean(cyto_fake, axis=-1)

        if verbose:
            print("Fake cyto shape:", cyto_fake.shape)

        row = roi_frame.loc[index]

        cyto_real_section = Section(os.path.join(test_folder, row.cyto_spline))
        cyto_real = cyto_real_section.image[margin:-margin, margin:-margin]

        if gray_scale:
            cyto_real = np.mean(cyto_real, axis=-1)

        if verbose:
            print("Real cyto shape:", cyto_real.shape)

        mse_result = mse(cyto_real, cyto_fake, mask=tissue_mask)
        rmse_result = rmse(cyto_real, cyto_fake, mask=tissue_mask)
        ssim_result = ssim(cyto_real, cyto_fake, mask=tissue_mask)
        mi_result = mi(cyto_real, cyto_fake, mask=tissue_mask)

        metrics[names[index]] = {
            'mse': mse_result,
            'rmse': rmse_result,
            'ssim': ssim_result,
            'mi': mi_result,
        }

    return metrics
