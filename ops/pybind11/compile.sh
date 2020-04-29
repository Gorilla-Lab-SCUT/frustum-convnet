#!/bin/bash
set -x 
set -e 

include=`python -m pybind11 --includes`

g++ -std=c++11 -shared -o box_ops_cc.so box_ops.cc -fPIC -O3 ${include}
g++ -std=c++11 -shared -o nms.so nms.cc -fPIC -O3 ${include}
