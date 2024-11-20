#!/bin/bash
#----------------------------------------------------
# HOME path on TACC is /home1/10287/amitayushthakur/
# WORK path on TACC is /work/10287/amitayushthakur/vista
# SCRATCH path on TACC is /scratch/10287/amitayushthakur/

#SBATCH -J proofwala_multilingual                                          # Job name
#SBATCH -o /scratch/10287/amitayushthakur/proofwala_multilingual.o%j       # Name of stdout output file
#SBATCH -e /scratch/10287/amitayushthakur/proofwala_multilingual.e%j       # Name of stderr error file
#SBATCH -p gh-dev                                                          # Queue (partition) name
#SBATCH -N 8                                                               # Total # of nodes (must be 1 for serial)
#SBATCH -n 8                                                               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00                                                        # Run time (hh:mm:ss)
#SBATCH --mail-type=all                                                    # Send email at begin and end of job
#SBATCH --mail-user=amitayush@utexas.edu

# Any other commands must follow all #SBATCH directives...
number_of_nodes=8 # Make sure to set this to the number of nodes you are using
batch_size=$1
if [ -z "$batch_size" ]; then
    batch_size=32
    echo "Batch size not provided. Setting to $batch_size."
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
per_node_batch_size=$((batch_size/number_of_nodes))
ibrun ./per_node_job.sh $per_node_batch_size $step_count $save_steps $eval_steps