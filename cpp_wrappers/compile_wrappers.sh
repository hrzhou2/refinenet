#!/bin/bash

cd cpp_normals
python3 setup.py build_ext --inplace
cd ..

cd cpp_height
python3 setup.py build_ext --inplace
cd ..
