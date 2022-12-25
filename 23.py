from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Iterable, NamedTuple

test_data = dedent(
    """
....#..
..###.#
#...#.#
.#...##
#.###..
##.#.##
.#..#..
""".strip()
).splitlines()

real_data = Path("inputs/23.txt").read_text().splitlines()
real = True

the_data = real_data if real else test_data


class Pos(NamedTuple):
    x: int
    y: int

    def __add__(self, other: "Pos") -> "Pos":
        return Pos(self.x + other.x, self.y + other.y)


def parse_input(lines: list[str]) -> set[Pos]:
    the_map = set()
    max_y = len(lines)
    max_x = len(lines[0])
    for y in range(max_y):
        line = lines[y]
        for x in range(max_x):
            if line[x] == "#":
                the_map.add(Pos(x, y))
    return the_map


def space_area(the_map: set[Pos]) -> int:
    min_x = min(_.x for _ in the_map)
    max_x = max(_.x for _ in the_map)
    min_y = min(_.y for _ in the_map)
    max_y = max(_.y for _ in the_map)
    return (max_y - min_y + 1) * (max_x - min_x + 1) - len(the_map)


deltas = [
    # North
    [Pos(-1, -1), Pos(0, -1), Pos(1, -1)],
    # South
    [Pos(-1, 1), Pos(0, 1), Pos(1, 1)],
    # West
    [Pos(-1, -1), Pos(-1, 0), Pos(-1, 1)],
    # East
    [Pos(1, -1), Pos(1, 0), Pos(1, 1)],
]
offsets = [_[1] for _ in deltas]
dirs = deque(range(4))
the_dirs = []
for _i in range(4):
    the_dirs.append(list(dirs))
    dirs.rotate(-1)


def get_neighbors(p: Pos, n: int) -> list[list[Pos]]:
    return [[p + delta for delta in deltas[i]] for i in the_dirs[n]]


def simulate(the_map: set[Pos], rounds: int, part: int = 0) -> int:
    def check_pos(ps: Iterable[Pos]) -> bool:
        for p in ps:
            if p in the_map:
                return False
        return True

    from_pos = []
    to_pos = []
    for _ in range(rounds):
        new_pos_cnt = defaultdict(int)
        round_mod = _ % 4
        from_pos.clear()
        to_pos.clear()
        for elf in the_map:
            checks = [check_pos(_) for _ in get_neighbors(elf, round_mod)[:4]]
            if all(checks):
                continue
            np = None
            _dirs = the_dirs[round_mod]
            for i, check in enumerate(checks):
                if check:
                    np = elf + offsets[_dirs[i]]
                    break
            if np is None:
                continue
            new_pos_cnt[np] += 1
            from_pos.append(elf)
            to_pos.append(np)
        moved = False
        for i, t in enumerate(to_pos):
            if new_pos_cnt[t] > 1:
                continue
            the_map -= {from_pos[i]}
            the_map |= {to_pos[i]}
            moved = True
        if not moved and part == 2:
            return _ + 1
    return space_area(the_map)


map1 = parse_input(the_data)
map2 = deepcopy(map1)

# Part 1
print(simulate(map1, 10))

# Part 2
print(simulate(map2, 200000, 2))
