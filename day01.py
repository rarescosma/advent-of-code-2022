import itertools
from pathlib import Path

lines = Path("inputs/01.txt").read_text().splitlines()

elves = [
    sum(map(int, _))
    for not_empty, _ in itertools.groupby(lines, key=bool)
    if not_empty
]

# Part 1
print(max(elves))
# Part 2
print(sum(sorted(elves)[-3:]))
