# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# this is a script for generating data for extrapolation task


import random
import numpy as np

MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"
END = "]"

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.25
MAX_ARGS = 5
MAX_DEPTH = 20
DATA_POINTS = 2000
MIN_LENGTH = 1000
MAX_LENGTH = 1100


def generate_tree(depth):
    if depth < MAX_DEPTH:
        r = random.random()
    else:
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value
    else:
        num_values = random.randint(2, MAX_ARGS)
        values = []
        for _ in range(num_values):
            values.append(generate_tree(depth + 1))

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t


def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'
        else:
            return to_string(t[0], parens) + ' ' + to_string(t[1], parens)


def to_value(t):
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return (np.sum(l[1]) % 10)
    elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])


with open(f"data/listops/external/test_{MIN_LENGTH}_{MAX_LENGTH}.tsv", 'w') as f:
    data = set()
    while len(data) < DATA_POINTS:
        example = generate_tree(1)
        if MIN_LENGTH < len(to_string(example, parens=False).split()) < MAX_LENGTH:
            data.add(example)
            # print(str(to_value(example)) + '\t' + to_string(example, parens=True))
            f.write(str(to_value(example)) + '\t' + to_string(example, parens=True) + '\n')
