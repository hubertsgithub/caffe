from celery import Celery
from lib.intrinsic.resulthandler import computeScoreJob_sendresults

app = Celery('comparison', backend='redis://:password@10.37.154.210:6379', broker='amqp://rabbitmqroot:password@10.187.16.216:44375')
#app = Celery('comparison', backend='redis://localhost', broker='amqp://')


@app.task
def computeScoreJob_task(*args, **kwargs):
    computeScoreJob_sendresults(*args, **kwargs)


@app.task
def print_task(*args, **kwargs):
    print args
    print kwargs

