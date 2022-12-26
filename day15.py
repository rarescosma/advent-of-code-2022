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

        for interval in sorted(self.intervals):
            if not merged:
                merged = [interval]
                continue

            if not Intervals.touches(interval, merged[-1]):
                merged.append(interval)
            else:
                merged[-1] = (merged[-1][0], max(interval[1], merged[-1][1]))

        return Intervals(merged)

    @staticmethod
    def touches(iv_a: Interval, iv_b: Interval) -> bool:
        return not (iv_a[1] < iv_b[0] - 1 or iv_b[1] < iv_a[0] - 1)

    def contains(self, val: int) -> bool:
        return any(_[0] <= val <= _[1] for _ in self.intervals)

    def __len__(self) -> int:
        return sum(_[1] - _[0] for _ in self.intervals)


@dataclass(frozen=True, order=True)
class Point:
    x_pos: int
    y_pos: int

    def manhattan(self, other: "Point") -> int:
        return abs(self.x_pos - other.x_pos) + abs(self.y_pos - other.y_pos)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x_pos + other.x_pos, self.y_pos + other.y_pos)


@dataclass
class Diag:
    # represents a line of the form y = x + intercept
    intercept: int

    @classmethod
    def from_point(cls, point: Point) -> "Diag":
        # intercept = -x + y
        return cls(-point.x_pos + point.y_pos)


@dataclass
class CounterDiag:
    # represents a line of the form y = -x + intercept
    intercept: int

    @classmethod
    def from_point(cls, point: Point) -> "CounterDiag":
        # intercept = x + y
        return cls(point.x_pos + point.y_pos)


def intersects(diag: Diag, counter: CounterDiag) -> list[Point]:
    # solve the system formed by the ecuations of the two diagonals:
    # 2*y = d.intercept + cd.intercept
    _y = (diag.intercept + counter.intercept) // 2
    _x = _y - diag.intercept

    # 45 degree integer-based lines can have a 4 point intersection
    # generate a grid of 3x3 points just to be safe
    deltas = [-1, 0, 1]
    return [Point(_x + dx, _y + dy) for (dx, dy) in product(deltas, deltas)]


def get_ints(line: str) -> list[int]:
    return [int(_) for _ in INT_RE.findall(line.strip())]


CONST_Y = 2000000
MAX_ROW = 4000000

sensors: list[tuple[Point, int]] = []
for _line in Path("inputs/15.txt").read_text().splitlines():
    (x_s, y_s, x_b, y_b) = get_ints(_line)
    sensor, beacon = Point(x_s, y_s), Point(x_b, y_b)
    sensors.append((sensor, sensor.manhattan(beacon)))


# x coordinates where a beacon cannot be on the given row
def row_intervals(row: int) -> Intervals:
    _intervals = Intervals([])
    for _sensor, _radius in sensors:
        _dy = abs(row - _sensor.y_pos)
        if _dy <= _radius:
            _dx = _radius - _dy
            _intervals.intervals.append(
                (_sensor.x_pos - _dx, _sensor.x_pos + _dx)
            )
    return _intervals.merge()


# Part 1
print(len(row_intervals(CONST_Y)))

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
    if 0 <= _.x_pos <= MAX_ROW and 0 <= _.y_pos <= MAX_ROW
}:
    intervals = row_intervals(candidate.y_pos)
    if not intervals.contains(candidate.x_pos):
        print(candidate.x_pos * 4000000 + candidate.y_pos)
        break
