#!/bin/bash

PASSWORD=$1

cd python/bkovacs

sudo rabbitmqctl status | grep "{rabbit,\"RabbitMQ\""
celery --version

sudo rabbitmqctl add_user rabbitmqroot $PASSWORD
sudo rabbitmqctl set_permissions rabbitmqroot ".*" ".*" ".*"
sudo rabbitmqctl list_users

celery flower -A lib.intrinsic.tasks --address=0.0.0.0 --port=5555
