echo "Setting up Thrall for training ..."
pushd ./imports/itp-interface
echo "Installing requirements for itp-interface..."
pip install -r requirements.txt
popd
echo "Building itp-interface..."
./src/proof_wala/scripts/update_imports.sh
echo "Installing thrall-lib requirements..."
pip install -r requirements.txt