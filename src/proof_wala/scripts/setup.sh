# if [[ ! -d "src/scripts" ]]; then
#     # Raise an error if the scripts directory is not present
#     echo "Please run this script from the root of the repository, cannot find src/scripts"
#     exit 1
# fi
# # Don't run without activating conda
# # Check if Conda is activated
# conda_status=$(conda info | grep "active environment" | cut -d ':' -f 2 | tr -d '[:space:]')
# if [[ $conda_status == "None" ]] || [[ $conda_status == "base" ]]; then
#     echo "Please activate conda environment before running this script"
#     exit 1
# fi
echo "Setting up Thrall ..."
echo "Installing OCaml (opam)..."
opam init -a --compiler=4.07.1
eval `opam config env`
opam update
# # For Coq:
echo "Installing Coq..."
opam pin add coq 8.10.2
opam pin -y add menhir 20190626
# # For SerAPI:
echo "Installing SerAPI (for interacting with Coq from Python)..."
opam install -y coq-serapi
echo "Installing Dpdgraph (for generating dependency graphs)..."
opam repo add coq-released https://coq.inria.fr/opam/released
opam install -y coq-dpdgraph
# python -m pip install -r requirements.txt
echo "Building Coq projects..."
(
    # Build CompCert
    echo "Building CompCert..."
    echo "This may take a while... (don't underestimate the time taken to build CompCert, meanwhile you can take a coffee break!)"
    pushd ./imports/itp-interface/src/data/benchmarks
    set -euv
    cd CompCert
    if [[ ! -f "Makefile.config" ]]; then
        ./configure x86_64-linux
    fi
    make -j `nproc`
    popd
    echo "CompCert built successfully!"
    # Ignore some proofs in CompCert
    # ./src/scripts/patch_compcert.sh
) || exit 1