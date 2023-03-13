# pip install livelossplot
pip install bposd json multiprocessing numpy torch
cat results.json || echo '{}' >> results.json
git pull --recurse-submodules
git submodule update --init --recursive
