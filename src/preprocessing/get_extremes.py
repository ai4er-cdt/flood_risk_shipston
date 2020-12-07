"""
Script assumes that the data is in the src/data directory.

python3 src/preprocessing/get_extremes.py
"""
import os
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import time
import matplotlib.pyplot as plt
from functools import wraps


def get_wiski_data(prefix, resolution="60min", field="stage", fill_in_value=np.nan):

    print("get wiski data script starting (takes a long time)")

    file_name = {
        "60min": prefix + "Shipston Wiski data - 60 min.xlsx",
        "15min": prefix + "Shipston Wiski data - 15 min.xlsx",
    }

    sheet_name = {"flow": 1, "stage": 2, "rainfall": 3}

    field_name = {
        "rainfall": "Precipitation [mm]",
        "stage": "Stage [m]",
        "flow": "Flow [m³/s]",
    }

    initial_dataframe = pd.read_excel(
        file_name[resolution],
        sheet_name=sheet_name[field],
        parse_dates=[["Date", "Time"]],
    )

    initial_dataframe["Date_Time"] = pd.to_datetime(initial_dataframe["Date_Time"])

    filtered_dataframe = initial_dataframe.replace(" ---", fill_in_value).replace(
        "  ---", np.nan
    )

    filtered_dataframe[field_name[field]] = pd.to_numeric(
        filtered_dataframe[field_name[field]], downcast="float"
    )

    return filtered_dataframe


def get_peaks_above_threshold(df, key, threshold):
    """
    A fairly naive but effective way of
    """
    npa = df[key].to_numpy()
    peaks, _ = find_peaks(npa, height=threshold)

    print("threshold", threshold, "peaks", len(peaks))

    return df.iloc[
        peaks.tolist(),
    ]


def output_excel_files(data_loc, stage60_filtered):
    for threshold in [2.7, 3.0, 3.4]:
        df = get_peaks_above_threshold(stage60_filtered, "Stage [m]", threshold)
        df.to_excel(data_loc + "stage_60min_" + str(threshold) + "m_threshold.xlsx")


def output_month_plots(PROJECT_PATH, stage60_filtered):
    for threshold in np.linspace(2.7, 3.5, 30):
        plt.figure()
        df = get_peaks_above_threshold(stage60_filtered, "Stage [m]", threshold)
        tmp_fig_path = os.path.join(PROJECT_PATH, "data", "tmp_images")
        if not os.path.exists(tmp_fig_path):
            os.mkdir(tmp_fig_path)
        df.groupby(df["Date_Time"].dt.month).count().plot(kind="bar")
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.savefig(
            os.path.join(
                tmp_fig_path, "stage_60min_" + str(threshold) + "m_threshold.png"
            ),
            dpi=500,
            bbox_inches="tight",
        )
        plt.clf()


if __name__ == "__main__":

    this_path = os.path.realpath(__file__)
    PROJECT_PATH = os.path.dirname(os.path.dirname(this_path))
    data_loc = os.path.join(PROJECT_PATH, "data") + "/"
    stage60_filtered = get_wiski_data(data_loc)

    # output_excel_files(data_loc, stage60_filtered)

    output_month_plots(PROJECT_PATH, stage60_filtered)
