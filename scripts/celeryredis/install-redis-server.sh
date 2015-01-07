#!/bin/bash

if [[ -d /redisdata ]]; then
    echo "Error: /redisdata already exists.  This script only works for the first setup"
    exit 1
fi
sudo apt-get -y update
sudo apt-get -y install python-dev python-pip redis-server git htop

sudo mkfs -t ext4 /dev/xvdb
sudo mkdir /data0
sudo mkdir /data0/redisdata
sudo bash -c 'echo "/dev/xvdb /data0 ext4 defaults,nofail,nobootwait 0 0" >> /etc/fstab'
sudo mount -a
sudo chown redis:redis /data0/redisdata
sudo service redis-server stop

echo "Now go and edit /etc/redis/redis.conf"
echo "Modify the working directory to \"/data0/redisdata\"!"
echo "Restart redis by \"redis-server /etc/redis/redis.conf\"!"
