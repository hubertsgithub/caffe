
def dicstr(dic, elemsep='-', keyvalsep='='):
    return elemsep.join([keyvalsep.join([str(k), str(v)]) for k, v in dic.iteritems()])

