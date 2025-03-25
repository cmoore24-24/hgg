#!/bin/bash

mamba activate coffea3 
python jet_processor.py --dataset $1
