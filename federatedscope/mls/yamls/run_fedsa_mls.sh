#!/bin/bash
#SBATCH --job-name=fedsa_lora_mls
#SBATCH --output=fedsa_lora_mls-%j.out
#SBATCH --error=fedsa_lora_mls-%j.err
#             #SBATCH --partition=gpubase_bygpu_b2  # allows 12 hours
#SBATCH --time=12:50:00           # Adjust time as needed
#                #SBATCH --partition=gpupreempt
#SBATCH --gres=gpu:h100:1              # Request 1 GPU
#SBATCH --cpus-per-task=4         # Adjust CPU cores
#SBATCH --mem=16G                 # Adjust memory

# Load necessary modules
#    module load StdEnv/2020
#       module load gcc/12.3
#    module load arrow/13.0.0
#    module load cuda/12.2
#    module load gcc arrow/18 python/3.10 rust cuda/12.2
module load python/3.10 arrow/18.1.0

# Activate your conda environment or virtualenv
source /home/ekhazaei/fedsa/bin/activate
             # Replace with your env name

#   #export HF_DATASETS_OFFLINE=1
#   #export TRANSFORMERS_OFFLINE=1
#   #export HF_HOME=$SCRATCH/hf_models
# Run your FedSA-LoRA training
python ./federatedscope/main.py --cfg ./federatedscope/mls/yamls/fedsa_lora_mls.yaml
