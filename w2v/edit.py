s = []
with open('scripts.txt') as f, open('w2v_scripts.txt', 'w') as w2v_f:
    idx = 0
    for l in f.readlines():
        s.append("%d\t%s" % (idx, l.split('\t')[1]))
        idx += 1

    w2v_f.writelines(s)
