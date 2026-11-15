# CHECK AND INSTALL LIBRARIES

import importlib.util
import subprocess
import sys

libraries = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "codecarbon": "codecarbon"
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

# MY FUNCTIONS 
from src.functions.metrics import my_metrics
from src.functions.cross_val_function import cross_validation_function

# ENERGY CONSUMPTION
from codecarbon import EmissionsTracker

import os        # gestione variabili d'ambiente
import pathlib   # gestire percorsi file 

powershell_path = (
    pathlib.Path(                                       # recupero directory principale di windoes 
        os.environ.get("SystemRoot", r"C:\Windows"))    # costruisce percorso di powershell 
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)

os.environ["PATH"] = (               # aggiungo cartella di powershell alla variabile PATH
    str(powershell_path.parent)
    + os.pathsep
    + os.environ.get("PATH", "")
)

print("PowerShell trovato:", powershell_path.exists())

# per code carbon voglio solo messaggi CRITICAL 
# altrimenti vengono dati in input messaggi INFO ogni secondo
import logging
logging.getLogger("codecarbon").setLevel(logging.CRITICAL)

print("All libraries successfully imported")

# due appunti su codecarbon fucntion
'''
tracker = EmissionsTracker(
    # nome del progetto
    project_name = "model_1",   

    # ambito del monitoraggio 
    # - "machine": considera consumo dell'intero sistama: CPU, RAM e GPU 
    # - "process": energia attribuibile al processo Python, utile quando lavoro ha altre esecuzioni in parallelo 
    # https://docs.codecarbon.io/latest/explanation/methodology/#tracking-modes
    tracking_mode = "process",                    

    # intrvallo di campionamento, ogni quanto viene stimato il consumo energetico, default 15
    # abbassare valore porta a osservare meglio variazioni temporali ma non garantisce stima più precisa
    # se troppo piccolo non potrebbe andare a interferire?
    measure_power_secs = 1, 

    # osservare solo info, warning o critical 
    # "INFO", "WARNING", "CRITICAL"
    log_level= "CRITICAL"
    
    # salvare risultati in un csv finale 
    # https://docs.codecarbon.io/latest/reference/output/
)

tracker.start()
# MODELLO 
tracker.stop()

# recupero valori utili
# https://docs.codecarbon.io/3.3/reference/output/ 
data = tracker.final_emissions_data
energy_wh = data.energy_consumed *1000    # default in kWh
duration_s = data.duration                # default in secondi 
emissions_g = data.emissions * 1000       # default in kg
'''
