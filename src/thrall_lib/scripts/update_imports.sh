#!/bin/bash

pushd imports/itp-interface
git submodule update --init --recursive
python -m pip install --upgrade build
python -m build
popd
pip install imports/itp-interface/dist/itp_interface-0.1.5-py3-none-any.whl --force-reinstall
install-itp-interface