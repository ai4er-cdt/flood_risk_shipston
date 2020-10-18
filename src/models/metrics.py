import numpy as np


def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliff-Efficiency.

    The NSE score is the standard in hydrology for assessing the predictive
    skill of models. 1 is perfect, 0 is rubbish!

    Args:
        obs (np.ndarray): Array containing the observations.
        sim (np.ndarray): Array containing the predictions.

    Returns:
        float: NSE value.
    """
    # only consider time steps where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator: float = np.sum((obs - np.mean(obs)) ** 2)
    numerator: float = np.sum((sim - obs) ** 2)
    nse_val: float = 1 - numerator / denominator

    return nse_val
