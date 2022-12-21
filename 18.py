from collections import deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from textwrap import dedent
from typing import Generator

real_data = Path("inputs/18.txt").read_text().splitlines()
test_data = dedent(
    """
2,2,2
1,2,2
3,2,2
2,1,2
2,3,2
2,2,1
2,2,3
2,2,4
2,2,6
1,2,5
3,2,5
2,1,5
2,3,5
""".strip()
).splitlines()


@dataclass(order=True, frozen=True)
class Cube:
    x: int
    y: int
    z: int

    def neighs(self) -> Generator["Cube", None, None]:
        for step, coord in product((-1, 1), range(3)):
            deltas = [0, 0, 0]
            deltas[coord] = step
            dx, dy, dz = deltas
            yield Cube(self.x + dx, self.y + dy, self.z + dz)

    def dim(self, d: int) -> int:
        if d == 0:
            return self.x
        if d == 1:
            return self.y
        if d == 2:
            return self.z
        raise ValueError(f"unknown dimension: {d}")

    def is_within(self, bot: "Cube", top: "Cube") -> bool:
        return all(bot.dim(_) <= self.dim(_) <= top.dim(_) for _ in range(3))


@dataclass(frozen=True)
class Face:
    top: Cube
    bot: Cube

    @classmethod
    def from_cubes(cls, c1: Cube, c2: Cube) -> "Face":
        return cls(top=c1, bot=c2) if c1 < c2 else cls(top=c2, bot=c1)


def common_faces(cs1: set[Cube], cs2: set[Cube]) -> set[Face]:
    return {
        Face.from_cubes(cube, neighbor)
        for cube in cs1
        for neighbor in set(cube.neighs()) & cs2
    }


def dfs_fill(droplet: set[Cube]) -> set[Cube]:
    bb_bot = Cube(*[min(_.dim(d) for _ in droplet) - 1 for d in range(3)])
    bb_top = Cube(*[max(_.dim(d) for _ in droplet) + 1 for d in range(3)])

    air = set()
    q = deque([bb_bot])

    while q:
        k = q.popleft()

        for neigh in k.neighs():
            if not neigh.is_within(bb_bot, bb_top):
                continue
            if neigh in air or neigh in droplet:
                continue
            air.add(neigh)
            q.append(neigh)

    return air


cubes = {Cube(*[int(_) for _ in line.split(",")]) for line in real_data}

# Part 1
print(6 * len(cubes) - 2 * len(common_faces(cubes, cubes)))

# Part 2

print(len(common_faces(cubes, dfs_fill(cubes))))
