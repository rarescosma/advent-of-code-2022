import re
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import reduce
from operator import add, mul
from pathlib import Path
from textwrap import dedent
from typing import Callable, Pattern

INT_RE: Pattern = re.compile("[^0-9,]+")

test_data = dedent(
    """
Monkey 0:
  Starting items: 79, 98
  Operation: new = old * 19
  Test: divisible by 23
    If true: throw to monkey 2
    If false: throw to monkey 3

Monkey 1:
  Starting items: 54, 65, 75, 74
  Operation: new = old + 6
  Test: divisible by 19
    If true: throw to monkey 2
    If false: throw to monkey 0

Monkey 2:
  Starting items: 79, 60, 97
  Operation: new = old * old
  Test: divisible by 13
    If true: throw to monkey 1
    If false: throw to monkey 3

Monkey 3:
  Starting items: 74
  Operation: new = old + 3
  Test: divisible by 17
    If true: throw to monkey 0
    If false: throw to monkey 1
    """.strip()
)

real_data = Path("inputs/11.txt").read_text()


@dataclass
class Monkey:
    items: deque[int]
    op: Callable[[int], int]
    div: int
    true_index: int
    false_index: int
    inspections: int = 0
    div_c: int = 0

    @classmethod
    def from_lines(cls, lines: list[str]) -> "Monkey":
        op_parts = lines[2].split(" ")
        term = op_parts[-1]
        _op = mul if op_parts[-2] == "*" else add

        return cls(
            items=deque(get_ints(lines[1])),
            op=lambda x: _op(x, x if term == "old" else int(term)),
            div=first_int(lines[3]),
            true_index=first_int(lines[4]),
            false_index=first_int(lines[5]),
        )

    def round(
        self,
        true_dest: "Monkey",
        false_dest: "Monkey",
        *,
        dampen: bool,
    ) -> None:
        while self.items:
            self.inspections += 1
            _item = self.op(self.items.popleft())
            if dampen:
                _item = _item // 3
            if self.div_c:
                _item %= self.div_c
            _dest = true_dest if _item % self.div == 0 else false_dest
            _dest.items.append(_item)


def get_ints(line: str) -> list[int]:
    nums = re.sub(INT_RE, "", line)
    if not nums:
        return []
    return [int(_) for _ in nums.split(",")]


def first_int(line: str) -> int:
    return next(iter(get_ints(line)))


def simulate_monkeys(
    monkeys: list[Monkey],
    rounds: int,
    dampen: bool = True,
) -> list[int]:
    for _ in range(rounds):
        for monkey in monkeys:
            monkey.round(
                monkeys[monkey.true_index],
                monkeys[monkey.false_index],
                dampen=dampen,
            )
    return sorted([_.inspections for _ in monkeys], reverse=True)


_lines = real_data.splitlines()
_monkeys = [
    Monkey.from_lines(_lines[offset : offset + 6])
    for offset in range(0, len(_lines), 7)
]

# add congruence factor - since we're only ever interested in the result
# of modulo operations of co-prime numbers (Monkey.div)
# we can constrain the value domain to a ring modulo their product
_div_c = reduce(mul, [_.div for _ in _monkeys])
_monkeys = [replace(_, div_c=_div_c) for _ in _monkeys]

# Part 1
p1 = simulate_monkeys(deepcopy(_monkeys), 20)
print(p1[0] * p1[1])

# Part 2
p2 = simulate_monkeys(deepcopy(_monkeys), 10000, dampen=False)
print(p2[0] * p2[1])
