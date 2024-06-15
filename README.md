# DROID-SLAM


<center><img src="misc/pipeline.png" width="1216" style="center"></center>



[DPT-SLAM: Dense Point Tracking for Visual SLAM](link_to_add_report)

Tjark Behrens, Damiano Da Col, Théo Ducrey and Wenqing Wang



**Initial Code Release:** This repo currently provides our implementation of DPT-SLAM as described in our report. This work was done in the context of the course 3D Vision at ETH Zürich


## Requirements

To run the code you will need ...
* **Inference:** Reproducing our run without training will require a GPU with at least 11G of memory. 

## Getting Started
1. Clone the repo using the `--recursive` flag
```Bash
git clone --recursive git@github.com:behretj/DOT-SLAM.git
```

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

[Optional] Set up custom modules from [PyTorch3D](https://github.com/facebookresearch/pytorch3d) to increase speed and reduce memory consumption of interpolation operations.
```
cd thirdparty/DOT/dot/utils/torch3d/ && python setup.py install && cd ../../..
```

3. Download the DOT's checkpoints
```
cd thirdparty/DOT/
mkdir checkpoints
cd checkpoints
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker_patch_4_wind_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_tapir.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_plus_bootstapir.pth
```

4. Download the DROID-SLAM's model from google drive: [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing)


## Demos

1. Download the scenes, we used for testing from polybox : 
```Bash
cd DOT-SLAM
mkdir datasets
# add here the directory TartanAir containing the scene from polybox
```

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


## Acknowledgements
Checkpoints : 
