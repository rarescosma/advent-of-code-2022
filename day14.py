from copy import deepcopy
from pathlib import Path
from textwrap import dedent

test_data = dedent(
    """
498,4 -> 498,6 -> 496,6
503,4 -> 502,4 -> 502,9 -> 494,9""".strip()
)

real_data = Path("inputs/14.txt").read_text()
real: bool = True

the_data = real_data if real else test_data


G = set()
floor: int = 0
ORIG: complex = 500

for line in the_data.splitlines():
    pairs = [list(map(int, p.split(","))) for p in line.split(" -> ")]

    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                G.add(x + y * 1j)
                floor = max(floor, y + 1)


def solve(part: int, grid: set[complex]) -> int:
    time = 0
    while part == 1 or ORIG not in grid:
        sand: complex = ORIG
        while True:
            if sand.imag >= floor:
                if part == 1:
                    return time
                break
            if sand + 1j not in grid:
                sand += 1j
            elif sand - 1 + 1j not in grid:
                sand += -1 + 1j
            elif sand + 1 + 1j not in grid:
                sand += 1 + 1j
            else:
                break
        grid.add(sand)
        time += 1
    return time


print(solve(1, deepcopy(G)))
print(solve(2, G))
