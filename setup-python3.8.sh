#!/bin/sh

apt-get install -y software-properties-common

# install PPA
add-apt-repository ppa:deadsnakes/ppa

# update and install
apt update
apt install -y python3.8 python3.8-dev python3.8-venv

# setup alternatives
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

# show menu for selecting the version
update-alternatives --config python3

curl https://bootstrap.pypa.io/get-pip.py | python3.8