#!/bin/bash

parent_path=$(dirname "${BASH_SOURCE[0]}")
echo "Script is located at $parent_path"
thrall_root=$(dirname $(dirname $(dirname $parent_path)))
echo "Changing directory to $thrall_root"
pushd $thrall_root
echo "Building thrall"
python3 -m pip install --upgrade build
python3 -m build
echo "Install thrall by referencing the wheel file in requirements.txt"
echo "You can refer to the wheel file in the dist directory in $thrall_root"
echo "For example, add the following line to your requirements.txt"
echo "$thrall_root/dist/proof_wala-0.0.2-py3-none-any.whl"
echo "Make sure to install itp_interface library as well, otherwise thrall will not work"
popd