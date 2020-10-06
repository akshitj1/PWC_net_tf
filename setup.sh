#!/bin/bash

# setup repo
python3 -m venv env
source env/bin/activate
pip install tensorflow
pip install tensorflow-addons
pip install pydot
pip install opencv-python