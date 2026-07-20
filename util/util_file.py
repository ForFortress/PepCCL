def toNum(l):
    l = [float(i) for i in l]
    return l
def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            l = line.split('\t')
            sequences.append(l[0])
            label = list(l[1])
            label = [int(i) for i in label]
            labels.append(label)
    return sequences, labels
