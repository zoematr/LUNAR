#!/usr/bin/env bash
#
#SBATCH --job-name routing-analysis
#SBATCH --output=/nfs/data8/matrullo/LUNAR/logs/routing_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH -p minor

WORK_DIR=/nfs/data8/matrullo/LUNAR
VENV_DIR=/nfs/data8/matrullo/venv_lunar

source $VENV_DIR/bin/activate
cd $WORK_DIR

python analyze_routing.py
