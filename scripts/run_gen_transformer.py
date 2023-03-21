from training_qec import main
import os
import sys
sys.path.append(os.getcwd() + "../src")
model = main(None, load_saved_scoring_model=False,
             skip_initialization_training=False, skip_eval=True, initialize_epoch_start=1, genetic_epoch_start=1)
