# ============================
#   IMPORTING LIBRARIES
# ======================================================================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from 

from rich.console import Console
console = Console()

# ============================
#   DATA LOADING
# ======================================================================================================================

train_df = pd.read_csv(r"Processed Datasets/final_train.csv")

test_df = pd.read_csv(r"Processed Datasets/final_train.csv")
