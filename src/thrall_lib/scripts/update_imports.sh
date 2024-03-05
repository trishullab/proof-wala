#!/bin/bash

git submodule update --init --recursive
pushd imports/itp-interface
python3 -m pip install --upgrade build
python3 -m build
popd