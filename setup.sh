#!/bin/bash

# setup repo
python3 -m venv env
source env/bin/activate
pip install tensorflow
pip install tensorflow-addons
pip install pydot
pip install opencv-python

# ref: https://janakiev.com/blog/jupyter-virtual-envs/
pip install --user ipykernel
python -m ipykernel install --user --name=tfenv
# uninstall
# jupyter kernelspec list
# jupyter kernelspec uninstall tfenv
