from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from operator import add, floordiv, mul, sub
from pathlib import Path
from textwrap import dedent
from typing import Union, cast

test_data = dedent(
    """
root: pppw + sjmn
dbpl: 5
cczh: sllz + lgvd
zczc: 2
ptdq: humn - dvpt
dvpt: 3
lfqf: 4
humn: 5
ljgn: 2
sjmn: drzm * dbpl
sllz: 4
pppw: cczh / lfqf
lgvd: ljgn * ptdq
drzm: hmdt - zczc
hmdt: 32
""".strip()
).splitlines()

real_data = Path("inputs/21.txt").read_text().splitlines()
real: bool = True

the_data = real_data if real else test_data


@dataclass
class Monkey:
    l_term: str
    oper: str
    r_term: str


Item = Union[Monkey, int]

OPS = {"*": mul, "/": floordiv, "+": add, "-": sub}
INV = {"*": "/", "/": "*", "+": "-", "-": "+"}
HUMN = "humn"

items: dict[str, Item] = {}
for line in the_data:
    parts = line.split(": ")
    if parts[1].isnumeric():
        items[parts[0]] = int(parts[1])
    else:
        subs = parts[1].split(" ")
        items[parts[0]] = Monkey(*subs[:3])


def solve(item_id: str, _items: dict[str, Item]) -> int:
    item = _items[item_id]
    if isinstance(item, int):
        return item
    return OPS[item.oper](
        solve(item.l_term, _items), solve(item.r_term, _items)
    )


def solve_equation(_items: dict[str, Item]) -> int:
    del _items[HUMN]
    _solve = partial(solve, _items=_items)

    # solve whatever we can, record "humn" dependencies
    overrides = {}
    humn_dep = set()
    for k in _items:
        try:
            overrides[k] = _solve(k)
        except KeyError:
            humn_dep.add(k)
    _items = {**_items, **overrides}

    # check for invariants:
    # 1) "humn" never appears as a RIGHT term
    # 2) root's RIGHT term doesn't depend on "humn" (so it solved as int)
    assert not any(isinstance(_, Monkey) and _.r_term == HUMN for _ in _items)
    assert (
        isinstance(_items["root"], Monkey)
        and _items["root"].r_term not in humn_dep
    )
    right = cast(int, _items[_items["root"].r_term])
    cur = _items[_items["root"].l_term]

    while isinstance(cur, Monkey) and cur.l_term != HUMN:
        # move cur.l_term to the other side
        if cur.r_term in humn_dep:
            # l_term * r_term = right => r_term = right / l_term
            # l_term / r_term = right => r_term = l_term / right
            # l_term + r_term = right => r_term = right - l_term
            # l_term - r_term = right => r_term = l_term - right
            assert isinstance(_items[cur.l_term], int)
            l_val = _solve(cur.l_term)
            if cur.oper == "*":
                right = right // l_val
            elif cur.oper == "/":
                right = l_val // right
            elif cur.oper == "+":
                right = right - l_val
            elif cur.oper == "-":
                right = l_val - right
            cur = _items[cur.r_term]
        # move cur.r_term to the other side
        elif cur.l_term in humn_dep:
            # l_term * r_term = right => l_term = right / r_term
            # l_term / r_term = right => l_term = right * r_term
            # l_term + r_term = right => l_term = right - r_term
            # l_term - r_term = right => l_term = right + r_term
            assert isinstance(_items[cur.r_term], int)
            r_val = _solve(cur.r_term)
            right = OPS[INV[cur.oper]](right, r_val)
            cur = _items[cur.l_term]

    assert isinstance(cur, Monkey)
    return OPS[INV[cur.oper]](right, _solve(cur.r_term))


# Part 1
print(solve("root", items))

# Part 2
humn = solve_equation(deepcopy(items))

items["humn"] = humn
assert isinstance(items["root"], Monkey)
assert solve(items["root"].l_term, items) == solve(items["root"].r_term, items)

print(humn)
