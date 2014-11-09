#!/bin/bash
# This script is useful to facilitate compiling on EC2 machines
# It creates a tmp directory on the drive which has more space than the default drive and changes TMPDIR to that directory

sudo mkdir -p /mnt/ubuntu/tmp
sudo chown -R ubuntu:ubuntu /mnt/ubuntu
export TMPDIR=/mnt/ubuntu/tmp
