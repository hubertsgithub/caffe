import multiprocessing
from multiprocessing.pool import Pool
import traceback


def call_with_multiprocessing_pool(func, *args):
    n_cpus = multiprocessing.cpu_count() - 1
    print "multiprocessing: using %s processes" % n_cpus
    pool = LoggingPool(processes=n_cpus)
    func(pool, *args)
    pool.close()
    pool.join()


# Shortcut to multiprocessing's logger
def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)


class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable
        return

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result
    pass


class LoggingPool(Pool):
    def apply_async(self, func, args=(), kwds={}, callback=None):
        return Pool.apply_async(self, LogExceptions(func), args, kwds, callback)
