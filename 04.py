from dataclasses import dataclass
from pathlib import Path

lines = Path("inputs/04.txt").read_text().splitlines()


@dataclass(frozen=True)
class Range:
    start: int
    end: int

    @classmethod
    def from_str(cls, x: str) -> "Range":
        range_ends = x.split("-")
        return cls(start=int(range_ends[0]), end=int(range_ends[1]))

    def is_valid(self) -> bool:
        return self.end >= self.start

    def contains(self, other: "Range") -> bool:
        return other.start >= self.start and other.end <= self.end

    def intersects(self, other: "Range") -> bool:
        # self.start            self.end
        #           other.start           other.end
        # => not completely to the left or right of other
        return not (self.end < other.start or other.end < self.start)


p1 = 0
p2 = 0
for line in lines:
    ends = line.split(",")
    r1, r2 = Range.from_str(ends[0]), Range.from_str(ends[1])
    if r1.contains(r2) or r2.contains(r1):
        p1 += 1
    if r1.intersects(r2):
        p2 += 1

print(p1)
print(p2)
