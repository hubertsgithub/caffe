import msgpack
import msgpack_numpy


def packb(x, version):
    """ Pack an object x (that can contain numpy objects) """
    return msgpack.packb({'version': version, 'data': x}, default=msgpack_numpy.encode)


def unpackb(x):
    """ Unpack an object x (that can contain numpy objects) """
    dic = msgpack.unpackb(x, object_hook=msgpack_numpy.decode)

    return dic['version'], dic['data']

