
BROKER_URL = 'amqp://rabbitmqroot:' + 'password' + '@10.239.170.36:5672'
CELERY_RESULT_BACKEND = 'redis://:' + 'password' + '@10.149.66.209:6379'

#CELERY_TIMEZONE = 'Europe/Oslo'
CELERY_TIMEZONE = 'UTC'
CELERY_ACKS_LATE = True
CELERYD_PREFETCH_MULTIPLIER = 1

