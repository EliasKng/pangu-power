#!/bin/bash
#SBATCH --partition=amd-milan-mi300
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --error=slogs/Error_%j.log
#SBATCH --output=slogs/finetune_%j.log
#SBATCH --mail-user=unfuq@student.kit.edu

module purge

source ./venv2/bin/activate

which python
echo "start"

python /hkfs/home/project/hk-project-test-mlperf/om1434/test_pg_pytorch_1/ma_repo/start_finetune_power.py --dist --gpu_list 0 1 2 3

echo "done"
