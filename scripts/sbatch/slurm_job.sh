#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=jw1524@ic.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/qdrl-robotics/qdax050/bin/:$PATH     # points to venv python path
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc
source activate
source /vol/cuda/12.0.0/setup.sh
# . /vol/cuda/12.0.0/setup.sh

# python ./main.py
python -c "import jax.numpy as jnp; x = jnp.array([1.0, 2.0, 3.0]); print(x.device)"
echo "%%%%%%%%%%%%%%%Running Complete%%%%%%%%%%%%%%%"