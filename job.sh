#!/bin/bash
#SBATCH --account=3dv
#SBATCH --nodes 1                  # 24 cores
#SBATCH --gpus 1
###SBATCH --gres=gpumem:24g
#SBATCH --time 00:30:00        ### adapt to our needs
#SBATCH --mem-per-cpu=12000
###SBATCH -J analysis1
#SBATCH -o job_output/dot-slam%j.out
#SBATCH -e job_output/dot-slam%j.err
###SBATCH --mail-type=END,FAIL

. /etc/profile.d/modules.sh
module load cuda/12.1
export CUB_HOME=/cluster/courses/3dv/data/team-4/DOT-SLAM/thirdparty/DOT/dot/utils/torch3d/cub-2.1.0
echo $CUB_HOME
export CXXFLAGS="-std=c++17"


echo "working"

export PYTHONPATH="/home/ddacol/DOT-SLAM/droid_slam/thirdparty/DOT"
echo $PYTHONPATH

source /cluster/courses/3dv/data/team-4/DOT-SLAM/env_dot-slam/bin/activate

# cd /cluster/courses/3dv/data/team-4/DOT-SLAM/
cd /home/ddacol/DOT-SLAM

#### put python commands here

./tools/validate_tartanair.sh --plot_curve

echo "finished"
