#!/usr/bin/env bash

echo "Installing cmake (requires password for sudo)"
sudo apt install cmake
echo "Installing face_recognition module, this might take a while..."
pip3 install face_recognition
echo "Pre-requirements installed."
