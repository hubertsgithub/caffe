
BROKER_URL = 'amqp://rabbitmqroot:' + 'password' + '@10.187.16.216:5672'
CELERY_RESULT_BACKEND = 'redis://:' + 'password' + '@10.37.154.210:6379'

#CELERY_TIMEZONE = 'Europe/Oslo'
CELERY_TIMEZONE = 'UTC'
CELERY_ACKS_LATE = True
CELERYD_PREFETCH_MULTIPLIER = 1

