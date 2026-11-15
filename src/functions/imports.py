# CHECK AND INSTALL LIBRARIES

import importlib.util
import subprocess
import sys

libraries = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "codecarbon": "codecarbon",
    "cpuinfo": "py-cpuinfo",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn"
}

for library, package in libraries.items():

    if importlib.util.find_spec(library) is None:
        print(f"Installing {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package
        ])

# GENERAL LIBRARIES
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# MACHINE LEARNING

from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    ShuffleSplit,
    StratifiedKFold,
    GridSearchCV,
    ParameterGrid,
    train_test_split
)

from sklearn.metrics import matthews_corrcoef

from sklearn.ensemble import RandomForestClassifier

# CPU INFORMATION

import cpuinfo

print("CPU model:", cpuinfo.get_cpu_info().get("brand_raw"))

# ENERGY CONSUMPTION

import os
import pathlib

# Add PowerShell to PATH

powershell_path = (
    pathlib.Path(os.environ.get("SystemRoot", r"C:\Windows"))
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)

if powershell_path.exists():
    os.environ["PATH"] = (
        str(powershell_path.parent)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

print("PowerShell found:", powershell_path.exists())

# CodeCarbon configuration
from codecarbon import EmissionsTracker

import logging
logging.getLogger("codecarbon").setLevel(logging.CRITICAL)

print("All libraries successfully imported")

# MY FUNCTIONS 
from src.functions.metrics import my_metrics
from src.functions.cross_val_function import cross_validation_function

# some notes on the CodeCarbon function
'''
tracker = EmissionsTracker(
    # project name
    project_name = "model_1",   

    # monitoring scope
    # - "machine": considers the consumption of the entire system: CPU, RAM and GPU
    # - "process": considers the energy attributable to the Python process, useful when other processes are running in parallel
    # https://docs.codecarbon.io/latest/explanation/methodology/#tracking-modes
    tracking_mode = "process",                    

    # sampling interval: how often energy consumption is estimated, default is 15
    # lowering this value allows more frequent observation of temporal variations but does not guarantee a more accurate estimate
    # if it is too small, could it interfere with the process?
    measure_power_secs = 1, 

    # display only info, warning or critical messages
    # "INFO", "WARNING", "CRITICAL"
    log_level= "CRITICAL"
    
    # save the final results to a CSV file
    # https://docs.codecarbon.io/latest/reference/output/
)

tracker.start()
# MODEL
tracker.stop()

# retrieve useful values
# https://docs.codecarbon.io/3.3/reference/output/
data = tracker.final_emissions_data
energy_wh = data.energy_consumed *1000    # default unit: kWh
duration_s = data.duration                # default unit: seconds
emissions_g = data.emissions * 1000       # default unit: kg
'''
