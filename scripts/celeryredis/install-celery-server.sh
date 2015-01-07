#!/bin/bash

PASSWORD=$1

if [[ -d /home/ubuntu/gitrepos ]]; then
    echo "Error: /home/ubuntu/gitrepos already exists.  This script only works for the first setup"
    exit 1
fi

echo "Adding line to /etc/apt/sources.list"
sudo bash -c 'echo "deb http://www.rabbitmq.com/debian/ testing main" >> /etc/apt/sources.list'
wget http://www.rabbitmq.com/rabbitmq-signing-key-public.asc
sudo apt-key add rabbitmq-signing-key-public.asc
sudo apt-get -y update
sudo apt-get -y install python-dev python-pip rabbitmq-server git htop
sudo pip install 'pip<1.6,>=1.5'
sudo pip install numpy
sudo pip install pypng pyamg redis msgpack-numpy Celery flower

mkdir /home/ubuntu/gitrepos
cd /home/ubuntu/gitrepos
git clone https://github.com/kovibalu/caffe.git
cd caffe
git checkout mitintrinsic
cd python/bkovacs

sudo rabbitmqctl status | grep "{rabbit,\"RabbitMQ\""
celery --version

sudo rabbitmqctl add_user rabbitmqroot $PASSWORD
sudo rabbitmqctl set_permissions rabbitmqroot ".*" ".*" ".*"
sudo rabbitmqctl list_users

celery flower -A lib.intrinsic.tasks --address=0.0.0.0 --port=5555
