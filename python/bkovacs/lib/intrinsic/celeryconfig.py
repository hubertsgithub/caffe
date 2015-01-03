
BROKER_URL = 'amqp://rabbitmqroot:' + 'password' + '@54.237.201.19:5672'
CELERY_RESULT_BACKEND = 'redis://:' + 'password' + '@54.145.153.202:6379'

#CELERY_TIMEZONE = 'Europe/Oslo'
CELERY_TIMEZONE = 'UTC'
CELERY_ACKS_LATE = True
CELERYD_PREFETCH_MULTIPLIER = 1

