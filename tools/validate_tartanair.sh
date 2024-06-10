#!/bin/bash


TARTANAIR_PATH=datasets/TartanAir_small

python evaluation_scripts/validate_tartanair.py --datapath=$TARTANAIR_PATH --disable_vis --buffer=400 $@
