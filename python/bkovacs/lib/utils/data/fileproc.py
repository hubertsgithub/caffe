
def freadlines(filepath, strip=True):
    with open(filepath, 'r') as f:
        if strip:
            lines = [s.strip() for s in f.readlines()]
        else:
            lines = f.readlines()

    return lines

