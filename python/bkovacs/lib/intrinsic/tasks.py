from celery import Celery

app = Celery('comparison', broker='amqp://guest@localhost//')

@app.task
def computeScoreJob_task(*args, **kwargs):
    from lib.intrinsic.comparison import computeScoreJob
    computeScoreJob(*args, **kwargs)


@app.task
def print_task(*args, **kwargs):
    print args
    print kwargs

