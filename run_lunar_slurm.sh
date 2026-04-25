#!/usr/bin/env bash
#
#SBATCH --job-name lunar-unlearn
#SBATCH --output=/nfs/data8/matrullo/LUNAR/logs/res_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH -p minor


# Paths
WORK_DIR=/nfs/data8/matrullo/LUNAR
VENV_DIR=/nfs/data8/matrullo/venv_lunar

# Activate virtual environment
source $VENV_DIR/bin/activate

# Move to project directory
cd $WORK_DIR

# Run LUNAR
python run_lunar.py
