from celery import Celery
from lib.intrinsic.resulthandler import computeScoreJob_sendresults

app = Celery('comparison', backend='redis://localhost', broker='amqp://')


@app.task
def computeScoreJob_task(*args, **kwargs):
    computeScoreJob_sendresults(*args, **kwargs)


@app.task
def print_task(*args, **kwargs):
    print args
    print kwargs
