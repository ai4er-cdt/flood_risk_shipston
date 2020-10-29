import numpy as np


def calc_nse(obs: np.ndarray, preds: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliff-Efficiency.

    The NSE score is the standard in hydrology for assessing the predictive
    skill of models. 1 is perfect, 0 is rubbish!

    Args:
        obs (np.ndarray): Array containing the observations.
        preds (np.ndarray): Array containing the predictions.

    Returns:
        float: NSE value.
    """
    # only consider time steps where observations are available
    preds = np.delete(preds, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    preds = np.delete(preds, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator: float = np.sum((obs - np.mean(obs)) ** 2)
    numerator: float = np.sum((preds - obs) ** 2)
    nse_val: float = 1 - numerator / denominator

    return nse_val
