import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Pattern, Tuple

INT_RE: Pattern = re.compile(r"-?\d+")
Interval = Tuple[int, int]


@dataclass
class Intervals:
    intervals: list[Interval]

    def merge(self) -> "Intervals":
        # N*log(N) consolidation of closed intervals
        # ex. (9, 11), (4, 7), (1, 3) -> (1, 7), (9, 11)
        merged: list[Interval] = []

        for iv in sorted(self.intervals):
            if not merged:
                merged = [iv]
                continue

            if not Intervals.touches(iv, merged[-1]):
                merged.append(iv)
            else:
                merged[-1] = (merged[-1][0], max(iv[1], merged[-1][1]))

        return Intervals(merged)

    @staticmethod
    def touches(a: Interval, b: Interval) -> bool:
        return not (a[1] < b[0] - 1 or b[1] < a[0] - 1)

    def contains(self, x: int) -> bool:
        return any(_[0] <= x <= _[1] for _ in self.intervals)

    def __len__(self) -> int:
        return sum(_[1] - _[0] for _ in self.intervals)


@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int

    def manhattan(self, o: "Point") -> int:
        return abs(self.x - o.x) + abs(self.y - o.y)

    def __add__(self, o: "Point") -> "Point":
        return Point(self.x + o.x, self.y + o.y)


@dataclass
class Diag:
    # represents a line of the form y = x + intercept
    intercept: int

    @classmethod
    def from_point(cls, p: Point) -> "Diag":
        # intercept = -x + y
        return cls(-p.x + p.y)


@dataclass
class CounterDiag:
    # represents a line of the form y = -x + intercept
    intercept: int

    @classmethod
    def from_point(cls, p: Point) -> "CounterDiag":
        # intercept = x + y
        return cls(p.x + p.y)


def intersects(d: Diag, cd: CounterDiag) -> list[Point]:
    # solve the system formed by the ecuations of the two diagonals:
    # 2*y = d.intercept + cd.intercept
    y = (d.intercept + cd.intercept) // 2
    x = y - d.intercept

    # 45 degree integer-based lines can have a 4 point intersection
    # generate a grid of 3x3 points just to be safe
    deltas = [-1, 0, 1]
    return [Point(x + dx, y + dy) for (dx, dy) in product(deltas, deltas)]


def get_ints(line: str) -> list[int]:
    return [int(_) for _ in INT_RE.findall(line.strip())]


const_y = 2000000
max_row = 4000000

sensors: list[tuple[Point, int]] = []
for _line in Path("inputs/15.txt").read_text().splitlines():
    (x_s, y_s, x_b, y_b) = get_ints(_line)
    sensor, beacon = Point(x_s, y_s), Point(x_b, y_b)
    sensors.append((sensor, sensor.manhattan(beacon)))


# x coordinates where a beacon cannot be on the given row
def row_intervals(row: int) -> Intervals:
    _intervals = Intervals([])
    for _sensor, _radius in sensors:
        dy = abs(row - _sensor.y)
        if dy <= _radius:
            dx = _radius - dy
            _intervals.intervals.append((_sensor.x - dx, _sensor.x + dx))
    return _intervals.merge()


# Part 1
print(len(row_intervals(const_y)))

# Part 2
diags = []
c_diags = []
for (sensor, radius) in sensors:
    # take the points just above / below the northern / southern tips of
    # the sensor's coverage
    np = sensor + Point(0, radius + 1)
    sp = sensor + Point(0, -radius - 1)

    # consider diagonals and counter-diagonals passing through them
    diags.extend([Diag.from_point(np), Diag.from_point(sp)])
    c_diags.extend([CounterDiag.from_point(np), CounterDiag.from_point(sp)])

# only consider points at the intersections of diagonal / counter-diagonal
# pairs that are also within bounds
for candidate in {
    _
    for _cross in product(diags, c_diags)
    for _ in intersects(*_cross)
    if 0 <= _.x <= max_row and 0 <= _.y <= max_row
}:
    intervals = row_intervals(candidate.y)
    if not intervals.contains(candidate.x):
        print(candidate.x * 4000000 + candidate.y)
        break
