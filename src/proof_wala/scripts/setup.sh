if [[ ! -d "src/proof_wala/scripts" ]]; then
    # Raise an error if the scripts directory is not present
    echo "Please run this script from the root of the repository, cannot find src/proof_wala/scripts"
    exit 1
fi
# Don't run without activating conda
# Check if Conda is activated
conda_status=$(conda info | grep "active environment" | cut -d ':' -f 2 | tr -d '[:space:]')
if [[ $conda_status == "None" ]] || [[ $conda_status == "base" ]]; then
    echo "Please activate conda environment before running this script"
    exit 1
fi
echo "Setting up ProofWala ..."
pushd ./imports/itp-interface
./src/itp_interface/scripts/setup.sh
popd
./src/proof_wala/scripts/update_imports.sh
pip_loc="./.conda/bin/pip"
$pip_loc install -r requirements.txt