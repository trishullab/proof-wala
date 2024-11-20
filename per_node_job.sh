#!/bin/bash -l
#----------------------------------------------------
# HOME path on TACC is /home1/10287/amitayushthakur/
# WORK path on TACC is /work/10287/amitayushthakur/vista
# SCRATCH path on TACC is /scratch/10287/amitayushthakur/
per_node_batch_size=$1
if [ -z "$per_node_batch_size" ]; then
    per_node_batch_size=8
    echo "Per node batch size not provided. Setting to $per_node_batch_size."
fi
step_count=$2
if [ -z "$step_count" ]; then
    step_count=100000
    echo "Step count not provided. Setting to $step_count."
fi
save_steps=$3
if [ -z "$save_steps" ]; then
    save_steps=10000
    echo "Save steps not provided. Setting to $save_steps."
fi
eval_steps=$4
if [ -z "$eval_steps" ]; then
    eval_steps=10000
    echo "Eval steps not provided. Setting to $eval_steps."
fi
module load gcc cuda/12.4
module load python3_mpi

if ! [ -d "$WORK/proofwala" ]; then
    echo "Directory $WORK/proofwala does not exist."
    echo "Exiting..."
    exit 1
fi
if ! [ -d "$WORK/thrall" ]; then
    echo "Directory $WORK/thrall does not exist."
    echo "Exiting..."
    exit 1
fi
if ! [ -d "$WORK/miniconda3" ]; then
    echo "Miniconda not found. Installing..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    echo "Installed Miniconda"
    source ~/.bashrc
fi
# pushd $WORK/thrall
# existed="True"
# if ! [ -d "$WORK/thrall/.conda" ]; then
#     echo "Creating conda environment..."
#     conda create -p ./.conda python=3.10
#     # python -m venv ./.conda
#     # source ./.conda/bin/activate
#     echo "Activating conda environment..."
#     conda activate ./.conda
#     python -m pip install --upgrade pip
#     python -m pip install torch==2.5.0.dev20240907+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
#     python -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name())"
#     output=$(python -c "import torch;print(torch.cuda.is_available())")
#     if [ "$output" = "False" ]; then
#         echo "CUDA is not available. Exiting..."
#         exit 1
#     fi
#     ./src/thrall_lib/scripts/setup_training.sh
#     existed="False"
# fi
# if [ $existed = "True" ]; then
#     echo "Activating conda environment..."
#     conda activate ./.conda
# fi
# CMD to Run
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eno1
export TRANSFORMERS_CACHE=$WORK/.cache/huggingface/hub
export ROOT="$WORK/proofwala"
pushd $WORK/thrall
echo "Activating conda"
conda activate ./.conda
torch_path=$(which torchrun)
echo "Torch path: $torch_path"
python_path=$(which python)
echo "Python path: $python_path"
./.conda/bin/python -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name())"
output=$(./.conda/bin/python -c "import torch;print(torch.cuda.is_available())")
if [ "$output" = "False" ]; then
    echo "CUDA is not available. Exiting..."
    exit 1
fi
slurm_out=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo "Slurm out: $slurm_out"
# Break slurm_out by newlines and convert into an array
IFS=$'\n' read -d '' -r -a nodelist <<< "$slurm_out"
# Get the number of nodes
num_nodes=${#nodelist[@]}
echo "Number of nodes: $num_nodes"
echo "Node list: ${nodelist[@]}"
# Get the master node
master_node=${nodelist[0]}
echo "Master node: $master_node"
# Get the master IP
master_ip=$(srun --nodes=1 --ntasks=1 -w $master_node hostname --ip-address)
echo "Master IP: $master_ip"
echo "Hostname: $(hostname)"
# Get the current node name
current_hostname=$(hostname)
# Remove the domain from the hostname
current_node=$(echo $current_hostname | cut -d'.' -f1)
# Calcluate the node rank based on the current node
node_rank=0
for node in "${nodelist[@]}"; do
    echo "Node: $node"
    if [ "$node" = "$current_node" ]; then
        break
    fi
    node_rank=$((node_rank+1))
done
echo "Node rank: $node_rank"
# Set the master port
master_port=61024
# Set the master address
master_addr=$master_node
./.conda/bin/torchrun \
    --nnodes $num_nodes \
    --node_rank $node_rank \
    --master_addr $master_addr \
    --master_port $master_port \
    src/thrall_lib/main/run.py --config-name coq_random_experiment training_settings.training_args.save_steps=$save_steps training_settings.training_args.eval_steps=$eval_steps training_settings.training_args.per_device_train_batch_size=$per_node_batch_size training_settings.training_args.max_steps=$step_count training_settings.training_args.seed=$RANDOM