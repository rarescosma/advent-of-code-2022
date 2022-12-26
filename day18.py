from collections import deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from textwrap import dedent
from typing import Any, Generator

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

real_data = Path("inputs/18.txt").read_text().splitlines()
real: bool = True

the_data = real_data if real else test_data


@dataclass(order=True, frozen=True)
class Cube:
    x_pos: int
    y_pos: int
    z_pos: int

    def neighs(self) -> Generator["Cube", None, None]:
        for step, coord in product((-1, 1), range(3)):
            deltas = [0, 0, 0]
            deltas[coord] = step
            _dx, _dy, _dz = deltas
            yield Cube(self.x_pos + _dx, self.y_pos + _dy, self.z_pos + _dz)

    def __getitem__(self, item: Any) -> int:
        if not isinstance(item, int):
            raise TypeError
        if item == 0:
            return self.x_pos
        if item == 1:
            return self.y_pos
        if item == 2:
            return self.z_pos
        raise ValueError(f"unknown dimension: {item}")

    def is_within(self, bot: "Cube", top: "Cube") -> bool:
        return all(bot[_] <= self[_] <= top[_] for _ in range(3))


@dataclass(frozen=True)
class Face:
    top: Cube
    bot: Cube

    @classmethod
    def from_cubes(cls, a_cube: Cube, b_cube: Cube) -> "Face":
        return (
            cls(top=a_cube, bot=b_cube)
            if a_cube < b_cube
            else cls(top=b_cube, bot=a_cube)
        )


def common_faces(cs1: set[Cube], cs2: set[Cube]) -> set[Face]:
    return {
        Face.from_cubes(cube, neighbor)
        for cube in cs1
        for neighbor in set(cube.neighs()) & cs2
    }


def dfs_fill(droplet: set[Cube]) -> set[Cube]:
    bb_bot = Cube(*[min(_[d] for _ in droplet) - 1 for d in range(3)])
    bb_top = Cube(*[max(_[d] for _ in droplet) + 1 for d in range(3)])

    air = set()
    queue = deque([bb_bot])

    while queue:
        k = queue.popleft()

        for neigh in k.neighs():
            if (
                not neigh.is_within(bb_bot, bb_top)
                or neigh in air
                or neigh in droplet
            ):
                continue
            air.add(neigh)
            queue.append(neigh)

    return air


cubes = {Cube(*[int(_) for _ in line.split(",")]) for line in the_data}

# Part 1
print(6 * len(cubes) - 2 * len(common_faces(cubes, cubes)))

# Part 2

print(len(common_faces(cubes, dfs_fill(cubes))))
