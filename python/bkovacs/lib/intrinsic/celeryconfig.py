from celeryconfig_local import PASSWORD, RABBITMQIP, REDISIP

BROKER_URL = 'amqp://rabbitmqroot:' + PASSWORD + '@' + RABBITMQIP + ':5672'
CELERY_RESULT_BACKEND = 'redis://:' + PASSWORD + '@' + REDISIP + ':6379'
#BROKER_URL = 'amqp://guest@localhost//'
#CELERY_RESULT_BACKEND = 'redis://localhost'

#CELERY_TIMEZONE = 'Europe/Oslo'
CELERY_TIMEZONE = 'UTC'
CELERY_ACKS_LATE = True
CELERYD_PREFETCH_MULTIPLIER = 1

