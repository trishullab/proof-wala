echo "Setting up Thrall for training ..."
echo "NOTE:This needs the virtual/conda env to be activated, and all git submodules to be updated."
if [[ ! -d "imports/itp-interface" ]]; then
    echo "Please run this script from the root of the repository, cannot find imports/itp-interface"
    echo "Make sure that imports/itp-interface submodule is updated."
    exit 1
fi
pushd ./imports/itp-interface
echo "Installing requirements for itp-interface..."
python -m pip install -r tacc_requirements.txt
echo "Building itp-interface..."
python -m pip install --upgrade build
python -m build
echo "itp-interface built successfully!"
popd
echo "Installing itp-interface..."
python -m pip install imports/itp-interface/dist/itp_interface-0.1.5-py3-none-any.whl --force-reinstall
echo "itp-interface installed successfully!"
echo "Installing thrall-lib requirements..."
python -m pip install -r tacc_requirements.txt
echo "thrall-lib requirements installed successfully!"