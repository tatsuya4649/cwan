#!/bin/bash

echo "training CWAN_L model"
python train_l.py -b 64 --start_epoch 9
echo "training CWAN_AB model"
python train_ab.py
