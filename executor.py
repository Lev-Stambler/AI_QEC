import os
import sys
sys.path.append(os.getcwd() + "/src")
from training import main
model = main(None, load_saved_scoring_model=True, skip_testing=True)