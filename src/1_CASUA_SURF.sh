#!/bin/zsh -f

python3 eval_topo_orders.py --II 33 --mmmt_model CASUA_SURF --DSP 360 --clock_period 8
python3 eval_topo_orders.py --II 33 --mmmt_model CASUA_SURF --DSP 840 --clock_period 8
python3 eval_topo_orders.py --II 33 --mmmt_model CASUA_SURF --DSP 1728 --clock_period 8

