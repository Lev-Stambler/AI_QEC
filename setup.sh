# Install Aff3ct
sudo add-apt-repository ppa:aff3ct/aff3ct-stable -y
sudo apt-get update -y
sudo apt-get install aff3ct-bin aff3ct-doc libaff3ct libaff3ct-dev -y

# Install Pip dependencies
pip install bposd numpy torch
cat results.json || echo '{}' >> results.json
