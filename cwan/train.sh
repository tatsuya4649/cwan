#!/bin/bash

echo "training CWAN_L model"
python train_l.py -b 128
echo "training CWAN_AB model"
python train_ab.py