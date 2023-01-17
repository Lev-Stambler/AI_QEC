import os
import sys
print("Starting run")
sys.path.append(os.getcwd() + "/src")
from training import main
model = main(None, load_saved_scoring_model=False, skip_testing=True)