
def freadlines(filepath, strip=True):
    with open(filepath, 'r') as f:
        if strip:
            lines = [s.strip() for s in f.readlines()]
        else:
            lines = f.readlines()

    return lines


def fwritelines(filepath, lines, endline=True):
    with open(filepath, 'w') as f:
        for l in lines:
            if endline:
                l = '{0}\n'.format(l)

            f.write(l)

