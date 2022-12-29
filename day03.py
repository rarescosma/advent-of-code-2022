from functools import reduce
from operator import and_
from pathlib import Path

lines = Path("inputs/03.txt").read_text().splitlines()


def to_prio(letter: str) -> int:
    big = ord("A")
    smol = ord("a")
    if ord(letter) >= smol:
        return ord(letter) - smol + 1
    return ord(letter) - big + 27


def to_prios(letters: str) -> set[int]:
    return {to_prio(letter) for letter in letters}


# Part 1
p1 = sum(
    (to_prios(line[len(line) // 2 :]) & to_prios(line[: len(line) // 2])).pop()
    for line in lines
)
print(p1)

# Part 2
sets = [to_prios(_) for _ in lines]
p2 = sum(
    reduce(and_, (sets[_] for _ in range(x, x + 3))).pop()
    for x in range(0, len(sets), 3)
)
print(p2)
