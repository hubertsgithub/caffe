#!/bin/bash

PASSWORD=$1

sudo rabbitmqctl status | grep "{rabbit,\"RabbitMQ\""
celery --version

sudo rabbitmqctl add_user rabbitmqroot $PASSWORD
sudo rabbitmqctl set_permissions rabbitmqroot ".*" ".*" ".*"
sudo rabbitmqctl list_users
