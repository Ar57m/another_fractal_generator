#!/bin/bash

rm *so
g++ -O3 -march=native -funroll-loops -ffast-math -fopenmp -shared -o libfract.so fract.cpp
python runner.py
