from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import NamedTuple

test_data = dedent(
    """
R 4
U 4
L 3
D 1
R 4
D 1
L 5
R 2
""".strip()
)

real_data = Path("inputs/09.txt").read_text()


class Pos(NamedTuple):
    x: int
    y: int

    def touching(self, other: "Pos") -> bool:
        return abs(self.x - other.x) <= 1 and abs(self.y - other.y) <= 1

    def step_towards(self, other: "Pos") -> "Pos":
        dx = self._towards(self.x, other.x)
        dy = self._towards(self.y, other.y)
        return Pos(dx, dy)

    @staticmethod
    def _towards(a: int, b: int) -> int:
        if a == b:
            return 0
        return 1 if b > a else -1

    def __add__(self, other: "Pos") -> "Pos":
        return Pos(self.x + other.x, self.y + other.y)


DELTAS: dict[str, Pos] = {
    "U": Pos(0, -1),
    "D": Pos(0, 1),
    "L": Pos(-1, 0),
    "R": Pos(1, 0),
}


@dataclass(frozen=True)
class Node:
    pos: Pos

    def move(self, delta: Pos) -> "Node":
        return Node(self.pos + delta)

    def catch_up(self, other: "Node") -> "Node":
        if other.pos.touching(self.pos):
            return self

        # catch up on both directions
        return self.move(self.pos.step_towards(other.pos))


head_pos = Pos(0, 0)
rope = [Node(head_pos)] * 10
seen = {head_pos}

for instr in real_data.splitlines():
    parts = instr.split(" ")
    (direction, steps) = parts[0], parts[1]

    _delta = DELTAS[direction]
    for _ in range(int(steps)):
        # move the head of the rope
        rope[0] = rope[0].move(_delta)

        # simulate following nodes
        for n in range(1, len(rope)):
            rope[n] = rope[n].catch_up(rope[n - 1])

        # record the tail position
        seen.add(rope[-1].pos)

print(len(seen))
