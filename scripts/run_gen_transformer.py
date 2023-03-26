import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd() + "/src")
from training_qec import main
model = main(None, load_saved_scoring_model=False,
             skip_initialization_training=False, skip_eval=True, initialize_epoch_start=1, genetic_epoch_start=1)
