#!/bin/bash


TARTANAIR_PATH=datasets/TartanAir

python evaluation_scripts/validate_tartanair.py --datapath=$TARTANAIR_PATH --weights=droid.pth --disable_vis --buffer=400 $@
#  --warmup=6
