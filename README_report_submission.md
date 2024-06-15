# DROID-SLAM


<img src="misc/pipeline.png" width="1216" style="center">



[DPT-SLAM: Dense Point Tracking for Visual SLAM](report.pdf)

Tjark Behrens, Damiano Da Col, Théo Ducrey and Wenqing Wang



**Initial Code Release:** This directory currently provides our implementation of DPT-SLAM as described in our report. This work was done in the context of the course 3D Vision at ETH Zürich
File structure of principal parts and contributions
DOT-SLAM\
├── datasets\
│   ├── TartanAir                               # Data :            Contain the different scenes we used for testing
│   │   ├── P000
│   │   ├── ...
│   │   └── P006
│   ├── TartanAir_small                         # Data :            Contain the different scenes we used for fast testing of new functions
├── evaluation_scripts/validate_tartanair.py    # Evaluation :      line 106 may be edited to switch testing between the different scene
├── droid_slam

├── thirdparty
│   ├── DOT
│   │   ├── Checkpoints
│   │   ├── dot
│   │   │   ├── models
│   │   │   │   ├── point_tracking.py           # Implementation :  implementing most of the changes of the section 3.2.1 of the report
├── tools/validate_tartanair.sh                 # Evaluation :      can be modified to switch testing between small scene and actual scene of Tatanair



## Requirements

* **Inference:** Reproducing our run without training will require a GPU with at least 11G of memory. 

## Getting Started
1. Download the complete project including the checkpoint, test_data, codes of each repositery directly from polybox :

2. Install dependencies

[Optional] Create a conda environment.
```
conda create -n dot python=3.9
conda activate dot
```

Install the [PyTorch and TorchVision](https://pytorch.org/get-started/locally/) versions which are compatible with your CUDA configuration.
```
pip install torch==2.3.0 torchvision==0.18.0
```

Install DOT inference dependencies.
```
pip install tqdm matplotlib einops einshape scipy timm lmdb av mediapy
```

DSet up custom modules from [PyTorch3D](https://github.com/facebookresearch/pytorch3d) to increase speed and reduce memory consumption of interpolation operations.
```
cd thirdparty/DOT/dot/utils/torch3d/ && python setup.py install && cd ../../..
```


## Demos

Run the demo on any of the samples (all demos can be run on a GPU with 11G of memory).

Python
```
Execute the termminal command of the job bellow
```

Sbatch
```
Create file job.sh with the content of the next section or replace the content of the existing example in the root directory of the repo
sbatch < job.sh
```


Job example (replace 'root_path' )
```
#!/bin/bash
#SBATCH --account=3dv
#SBATCH --nodes 1                  # 24 cores
#SBATCH --gpus 1
###SBATCH --gres=gpumem:24g
#SBATCH --time 02:00:00        ### adapt to our needs
#SBATCH --mem-per-cpu=12000
###SBATCH -J analysis1
#SBATCH -o job_output/dot-slam%j.out
#SBATCH -e job_output/dot-slam%j.err
###SBATCH --mail-type=END,FAIL

. /etc/profile.d/modules.sh
module load cuda/12.1
export CUB_HOME=$root_path$/DOT-SLAM/thirdparty/DOT/dot/utils/torch3d/cub-2.1.0
echo $CUB_HOME
export CXXFLAGS="-std=c++17"

echo "working"
export PYTHONPATH="$root_path$/DOT-SLAM/droid_slam/thirdparty/DOT"
echo $PYTHONPATH
source $root_path$/DOT-SLAM/env_dot-slam/bin/activate
cd $root_path$/DOT-SLAM
./tools/validate_tartanair.sh --plot_curve
echo "finished"
```

## Evaluation




