""" Utilies related to performing batch processing  """

from progressbar import (Bar, ETA, AdaptiveETA, FileTransferSpeed, ProgressBar,
                         SimpleProgress)


def progress_bar(l, show_progress=True, adaptive=True):
    """ Returns an iterator for a list or queryset that renders a progress bar
    with a countdown timer """

    if show_progress:
        return iterator_progress_bar(l, adaptive=adaptive)
    else:
        return l


def progress_bar_widgets(adaptive=True):
    if adaptive:
        eta = AdaptiveETA()
    else:
        eta = ETA()

    return [
        SimpleProgress(sep='/'), ' ', Bar(), ' ',
        FileTransferSpeed(unit='items'), ', ', eta
    ]


def iterator_progress_bar(iterator, maxval=None, adaptive=True):
    """ Returns an iterator for an iterator that renders a progress bar with a
    countdown timer """

    if maxval is None:
        try:
            maxval = len(iterator)
        except:
            return iterator

    if maxval > 0:
        pbar = ProgressBar(maxval=maxval, widgets=progress_bar_widgets(adaptive=adaptive))
        return pbar(iterator)
    else:
        return iterator


def queryset_progress_bar(queryset):
    """ Returns an iterator for a queryset that renders a progress bar with a
    countdown timer """
    return iterator_progress_bar(queryset.iterator(), maxval=queryset.count())


def iter_batch(iterable, n=1024):
    """ Group an iterable into batches of size n """
    if hasattr(iterable, '__getslice__'):
        for i in xrange(0, len(iterable), n):
            yield iterable[i:i+n]
    else:
        l = []
        for x in iterable:
            l.append(x)
            if len(l) >= n:
                yield l
                l = []
        if l:
            yield l


def group_iterable_by_attr(iterable, attr):
    """ Returns a dictionary mapping
    { val : [elements such that e.attr = val] } """
    groups = {}
    for e in iterable:
        val = getattr(e, attr)
        if val in groups:
            groups[val].append(e)
        else:
            groups[val] = [e]
    return groups

