#!/bin/bash

nohup scp -i ~/.ssh/auth -r ubuntu@54.176.3.57:/home/ubuntu/Projects/data/proofsteps/training_results/models/thrall-codet5-base-coq-lean-4-2048/checkpoint-2* /mnt/sdd1/amthakur/foundry/models2/thrall-codet5-base-coq-lean-4-2048/ &
nohup scp -i ~/.ssh/auth -r ubuntu@54.176.3.57:/home/ubuntu/Projects/data/proofsteps/training_results/models/thrall-codet5-base-mathlib-2048 /mnt/sdd1/amthakur/foundry/models 