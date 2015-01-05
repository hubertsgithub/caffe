import msgpack
import msgpack_numpy


def packb(x, version):
    """ Pack an object x (that can contain numpy objects) """
    return msgpack.packb({'version': version, 'data': x}, default=msgpack_numpy.encode)


def fpackb(x, version, filepath):
    """ Pack an object x and save it to file (that can contain numpy objects) """
    packed = packb(x, version)

    with open(filepath, 'w') as f:
        f.write(packed)


def unpackb(x):
    """ Unpack an object x (that can contain numpy objects) """
    dic = msgpack.unpackb(x, object_hook=msgpack_numpy.decode)

    return dic['version'], dic['data']


def unpackb_version(packed, expected_version):
    """ Unpack an object x (that can contain numpy objects) """
    dic = msgpack.unpackb(packed, object_hook=msgpack_numpy.decode)
    package_version = dic['version']

    if package_version != expected_version:
        raise ValueError('Unexpected version ({0}) when reading package (expected: {1})'.format(package_version, expected_version))

    return dic['data']


def funpackb_version(expected_version, filepath):
    """ Unpack an object x from file (that can contain numpy objects) """
    with open(filepath, 'r') as f:
        packed = f.read()

    return unpackb_version(packed, expected_version)
