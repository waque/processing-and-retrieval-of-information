#!/usr/bin/env python3

from   table import *

import argparse
import pprint

def parse():
    parser = argparse.ArgumentParser(description='Compute query document(s) similarity')
    parser.add_argument('docindex', type=str, help='file with documents\' names')
    parser.add_argument('--query', type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    docnames = parse().docindex
    q = parse().query
    t = table(docnames)

    pprint.pprint(sorted(similarity(t, q).items(), key = lambda x : x[1], reverse=True))
