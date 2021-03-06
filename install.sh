#!/bin/bash
#setting pip and making virtual python environments
#and python libraries install

sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -r build-essential python3-pip python3-venv
sudo ln -s /usr/bin/pip3 /usr/bin/pip # to move pip command

sudo ln -s /usr/bin/python3 /usr/bin/python # to move python command

echo "install the required libraries..."
echo "================================="
pip install -U pip
if pip install -r requirements.txt; then
	echo "ok, success"
fi
