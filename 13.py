from functools import cmp_to_key, reduce
from operator import mul
from pathlib import Path
from textwrap import dedent
from typing import Union

real_data = Path("inputs/13.txt").read_text()

test_data = dedent(
    """
[1,1,3,1,1]
[1,1,5,1,1]

[[1],[2,3,4]]
[[1],4]

[9]
[[8,7,6]]

[[4,4],4,4]
[[4,4],4,4,4]

[7,7,7,7]
[7,7,7]

[]
[3]

[[[]]]
[[]]

[1,[2,[3,[4,[5,6,7]]]],8,9]
[1,[2,[3,[4,[5,6,0]]]],8,9]
""".strip()
)


def comp_terms(left_l: Union[int, list], right_l: Union[int, list]) -> int:
    if isinstance(left_l, int):
        if isinstance(right_l, list):
            return comp_terms([left_l], right_l)
        else:
            return left_l - right_l
    elif isinstance(right_l, int):
        return comp_terms(left_l, [right_l])

    for left_i, right_i in zip(left_l, right_l):
        if _cmp := comp_terms(left_i, right_i):
            return _cmp
    return len(left_l) - len(right_l)


# Part 1
p1 = 0
packets = []
for _i, pairs in enumerate(real_data.split("\n\n")):
    parts = pairs.splitlines()
    left_term, right_term = map(eval, parts)
    packets.extend([left_term, right_term])
    if comp_terms(left_term, right_term) < 0:
        p1 += _i + 1
print(p1)

# Part 2
markers = [[[2]], [[6]]]
packets.extend(markers)

packets = sorted(packets, key=cmp_to_key(comp_terms))
p2 = reduce(mul, [packets.index(_) + 1 for _ in markers])
print(p2)
