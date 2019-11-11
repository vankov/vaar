import numpy as np
max_el = 0
min_el = 0

with open("glove.50d.embeddings.txt", "r") as F:
    for line in F:
        els = map(lambda(x): float(x), line.split(" ")[1:])
        m1 = np.max(els)
        m2 = np.min(els)
        if (m1 > max_el) or (m2 < min_el):
            max_el = max(max_el, m1)
            min_el = min(min_el, m2)
            print("{}\t{}".format(min_el, max_el))