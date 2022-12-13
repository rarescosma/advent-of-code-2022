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

    def intersect(self, other: "Range") -> "Range":
        return Range(
            start=max(self.start, other.start),
            end=min(self.end, other.end),
        )

    def overlaps(self, other: "Range") -> bool:
        return self.intersect(other).is_valid()


acc = 0
ol = 0

for line in lines:
    ends = line.split(",")
    r1, r2 = Range.from_str(ends[0]), Range.from_str(ends[1])
    if r1.contains(r2) or r2.contains(r1):
        acc += 1
    if r1.overlaps(r2):
        ol += 1

print(acc)
print(ol)
