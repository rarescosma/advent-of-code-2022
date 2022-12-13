from functools import cmp_to_key, reduce
from operator import mul
from pathlib import Path
from textwrap import dedent
from typing import Any

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


def _is_int(thing: Any) -> bool:
    return not isinstance(thing, list)


def comp_lists(left_l: list, right_l: list) -> int:
    upper = min(len(left_l), len(right_l))
    i = 0
    while i < upper:
        left_i, right_i = left_l[i], right_l[i]
        left_i = [left_i] if _is_int(left_i) else left_i
        right_i = [right_i] if _is_int(right_i) else right_i

        if cmp := (
            comp_ints(left_i[0], right_i[0])
            if len(left_i) == len(right_i) == 1
            and _is_int(left_i[0])
            and _is_int(right_i[0])
            else comp_lists(left_i, right_i)
        ):
            return cmp
        i += 1
    return comp_ints(len(left_l[i:]), len(right_l[i:]))


def comp_ints(left: int, right: int) -> int:
    return 1 if left < right else -1 if left > right else 0


# Part 1
p1 = 0
for _i, pairs in enumerate(real_data.split("\n\n")):
    parts = pairs.splitlines()
    left_list = eval(parts[0])
    right_list = eval(parts[1])
    if res := comp_lists(left_list, right_list) == 1:
        p1 += _i + 1
print(p1)

# Part 2
packets = [eval(p) for p in real_data.splitlines() if p]
markers = [[[2]], [[6]]]
packets.extend(markers)

packets = sorted(packets, key=cmp_to_key(comp_lists), reverse=True)
p2 = reduce(mul, [packets.index(_) + 1 for _ in markers])
print(p2)
