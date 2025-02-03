# ProofWala

## Introduction
Neural networks have shown substantial promise at automatic theorem-proving in interactive proof assistants (ITPs) like Lean and Coq. However, most neural theorem-proving models are restricted to specific ITPs, leaving out opportunities for cross-lingual transfer between ITPs. We address this weakness with a multilingual proof framework, <span style="font-variant:small-caps;">ProofWala</span>, that allows a standardized form of interaction between neural theorem-provers and two established ITPs (Coq and Lean). It enables the collection of multilingual proof step data---data recording the result of proof actions on ITP states---for training neural provers. 
 <span style="font-variant:small-caps;">ProofWala</span> allows the systematic evaluation of a model's performance across different ITPs and problem domains via efficient parallel proof search algorithms. We show that multilingual training enabled by <span style="font-variant:small-caps;">ProofWala</span> can lead to successful transfer across ITPs. 

 ## Framework Details
 <span style="font-variant:small-caps;">ProofWala</span>, a unified framework for extracting and organizing training data for formal theorem proving in <span style="font-variant:small-caps;">Lean</span> and <span style="font-variant:small-caps;">Coq</span>. The framework supports the generation of training data for these ITPs from any formal theorem-proving Git repository (such as Mathlib, CompCert, MathComp, GeoCoq, & Category Theory) and the subsequent training of LLMs for single-proof step generation. Our data collection format is standardized across ITPs, and we have created generic prompt formatting schemes across different ITPs and domains. The framework also helps with end-to-end proof search and collection of the annotated proof trees which can be further used for analysis and visualization.

 Our framework has three components: 
(i) the **interface module**: for the execution of proof steps (tactics) on various ITPs, (ii) the **proof step generation & training module**: for generating proof step data and training proof step generation model, and (iii) the **parallel proof search module**: for using the guidance from proof step generation model to do end-to-end the proof search. The Figure below shows the interaction between our different modules.

## Installation
1. Create a conda environment using the following command:
```bash
conda create -p ./.conda python=3.10
```
2. Activate the conda environment:
```bash
conda activate ./.conda
```
3. Get all code dependencies from the submodules:
```bash
git submodule update --init --recursive
```
4. Install Coq:

    a. Install OCaml first. Use the instructions here: https://opam.ocaml.org/doc/Install.html. The opam version used in this project is 2.1.3 (OCaml 4.14.0). This can be changed to any other version too, a good way to do that will be using switches. Note that OCaml officially only supports Linux installations. One can use WSL on Windows machines.

    b. Run the following to install Coq on Linux. The Coq version supported in this project is <= 8.18 and >= 8.10. 
    ```
    sudo apt install build-essential unzip bubblewrap
    sh <(curl -sL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)
    ```

    c. Add the following to your `.bashrc` file: (sometimes the path `~/.opam/default` might not exist, so use the directory with version number present in the `~/.opam` directory)
    ```
    export PATH="/home/$USER/.opam/default/bin:$PATH"
    ```

4. Install Lean and build all dependencies:
```bash
pushd imports/itp-interface
./src/itp_interface/scripts/setup.sh
# The setup script will install Lean version 4.7.0-rc2 by default
# If you want a different version, then you will have to follow steps mentioned in installing differnt Lean Version section
popd
```

5. Install the Python dependencies for training the models:
```bash
./src/scripts/setup_training.sh
```

6. To generate the code for the proof step data generation, run the following command from the root directory:
```bash
pushd imports/itp-interface
python src/itp_interface/main/run_tool.py --config-name simple_coq_data_gen
popd
```
>Note: Check the `simple_coq_data_gen.yaml` configuration in the `imports/itp-interface/src/itp_interface/configs` directory for more details.

7. To train the model for proof step generation, run the following command from the root directory:
```bash
torchrun --nproc-per-node 2 --master-port 31052 src/proof_wala/main/run.py --config-name experiment
# ^ This will run training on 2 GPUs, on the same node
# ^ For a single node or no GPU training just remove the --nproc-per-node 2 and --master-port 31052 and torchrun
# The above training job can also be run on a slurm cluster/(or any distributed cluster), for that refer the per_node_job.sh and tacc_slurm.sh script in the root directory
```
>Note: Check the `experiment.yaml` configuration in the `src/proof_wala/configs` directory for the exact details of the training configuration and where the model will be saved, the location where it expects the data to be present, etc.

8. Install the dependencies for the parallel proof search module:
```bash
pushd imports/itp-interface
python -m pip install --upgrade build
python -m build
popd
pip install imports/itp-interface/dist/itp_interface-0.1.5-py3-none-any.whl --force-reinstall
install-itp-interface
pip install -r requirements.txt
```

9. To run the parallel proof search module, run the following command from the root directory:
```bash
export FOLLOW_SEED="True"
export ROOT="<path to the directory where the models are saved>"
# The models are assumed to be stored in the directory <ROOT>/models/<model_name>
# ^ You can change these paths from the YAML files in the src/proof_wala/configs directory
export CUDA_VISIBLE_DEVICES="1,2,3,4" # Based on the number of GPUs available
# Start the ray cluster
mkdir -p .log/ray
python src/proof_wala/main/init.py &
# ^ This will start the ray cluster in the background
# ^ a .log/ray/session_latest file which will have the details of the ray cluster
# ^ you can also create the session file with info of an existing ray cluster without starting a new one
python src/proof_wala/main/run_proof_search.py --config-name eval_simple_lean_test_multilingual
#^ This will automatically start the proof search on the ray cluster mentioned in the session_latest file
#^ For a distributed run we assume that each node has access to the same data, models, essentially the same file system (NFS, SMB, etc)
```
>Note: for the `session_latest` file, we use the following json format:
```json
{
    "node_ip_address": "127.0.0.132", 
    "raylet_ip_address": "127.0.0.132", 
    "redis_address": null, 
    "object_store_address": "/tmp/ray/session_2025-01-30_18-30-47_764457_139622/sockets/plasma_store", 
    "raylet_socket_name": "/tmp/ray/session_2025-01-30_18-30-47_764457_139622/sockets/raylet", "webui_url": "127.0.0.1:8265", 
    "session_dir": "/tmp/ray/session_2025-01-30_18-30-47_764457_139622",
    "metrics_export_port": 64063, 
    "gcs_address": "127.0.0.132:61362", 
    "address": "127.0.0.132:61362",
    "dashboard_agent_listen_port": 52365, 
    "node_id":"1d44a4b8ce6d2bbf51077e7289b648315c291bbaba1d7dc9aa4c6497", 
    "main_pid": 139622
}
```
