import os
import re

def read_file(path):
    if path.endswith(".json"):
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            contents = contents.replace('<', '[').replace('>', ']')#.replace('"',"^")
            lines = [line.strip() for line in contents.split('\n')]
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines
def write_file(write_path, result):
    with open(write_path, 'w') as f:
        f.write('\n'.join(result))

def split_triples(text):
    triples, triple = [], []
    for w in text:
        if w not in ['[TRIPLE]', '[/TRIPLE]']:
            triple.append(w)
        elif w == '[/TRIPLE]':
            triples.append(triple)
            triple = []
    return triples


def join_triples(triples):
    result = []
    for triple in triples:
        result.append('[TRIPLE]')
        result.extend(triple)
        result.append('[/TRIPLE]')
    return result


def orderout2structin(ordering_out, triples):
    ord_triples = []
    if len(triples) == 1:
        ord_triples.extend(triples)
    else:
        added = []
        for predicate in ordering_out:
            for i, triple in enumerate(triples):
                if predicate.strip() == triple[1].strip() and i not in added:
                    ord_triples.append(triple)
                    added.append(i)
                    break
    return ' '.join(join_triples(ord_triples))

# EDIT THIS CODE FROM HERE!!!!
def structout2lexin(struct_out, triples):
    sentences, snt = [], []
    for w in struct_out:
        if w.strip() not in ['[SNT]', '[/SNT]']:
            snt.append(w.strip())

        if w.strip() == '[/SNT]':
            sentences.append(snt)
            snt = []

    struct, struct_unit = [], []
    if len(triples) == 1:
        struct.append(triples)
    else:
        added = []
        for snt in sentences:
            for predicate in snt:
                for i, triple in enumerate(triples):
                    if predicate.strip() == triple[1].strip() and i not in added:
                        struct_unit.append(triple)
                        added.append(i)
                        break
            struct.append(struct_unit)
            struct_unit = []
    return ' '.join(utils.join_struct(struct))

def lexout2regin(lex_out, triples):
    entities = utils.entity_mapping(triples)
    for i, w in enumerate(lex_out):
        if w.strip() in entities:
            lex_out[i] = entities[w]
    return ' '.join(lex_out)

# STOP HERE!!!!

def run(out_path, entries_path):
    outputs = read_file(out_path)
    outputs = [out.split() for out in outputs]

    entries = read_file(entries_path)
    entries = [split_triples(t.split()) for t in entries]

    for i, entry in enumerate(entries):
        yield orderout2structin(ordering_out=outputs[i], triples=entry)


# Receive entry path from Data/results
entries_path_dev = "pipeline_datasets/dev.eval"
out_path_dev = "pipeline_results/dev.ordering"

entries_path_test = "pipeline_datasets/test.eval"
out_path_test = "pipeline_results/test.ordering"

write_path_dev = "pipeline_results/dev.ordering.mapped"
write_path_test = "pipeline_results/test.ordering.mapped"

result_dev = run(out_path=out_path_dev, entries_path=entries_path_dev)
result_test = run(out_path=out_path_test, entries_path=entries_path_test)


write_file(write_path_dev, result_dev)
write_file(write_path_test, result_test)


