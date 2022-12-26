import bisect
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from textwrap import dedent
from typing import Generator, NamedTuple

HeightMap = dict[int, dict[int, list[int]]]
MAX_HEIGHT: int = 9

test_data = dedent(
    """
30373
25512
65332
33549
35390
""".strip()
)

real_data = Path("inputs/08.txt").read_text()


class Pos(NamedTuple):
    x: int
    y: int


Size = Pos


class TreeStats(NamedTuple):
    visible: bool
    scenic_score: int


@dataclass(frozen=True)
class Map:
    lines: list[str]
    row_map: HeightMap
    col_map: HeightMap
    size: Size

    @classmethod
    def from_lines(cls, lines: list[str]) -> "Map":
        max_y = len(lines)
        max_x = len(lines[0])

        # row by y, col by x
        row_map = default_map()
        col_map = default_map()

        part = cls(lines, {}, {}, Size(max_x, max_y))

        for (x, y) in part.iter():
            el = part.get(x, y)
            row_map[el][y].append(x)
            col_map[el][x].append(y)

        return replace(part, row_map=row_map, col_map=col_map)

    def get(self, x: int, y: int) -> int:
        return int(self.lines[y][x])

    def stats(self, x: int, y: int) -> TreeStats:
        if x == 0 or y == 0 or x == self.size.x - 1 or y == self.size.y - 1:
            return TreeStats(True, 0)

        height = self.get(x, y)
        vis_w, vis_e, vis_n, vis_s = True, True, True, True
        box_w, box_e, box_n, box_s = (
            0,
            self.size.x - 1,
            0,
            self.size.y - 1,
        )

        for target_height in range(height, MAX_HEIGHT + 1):
            if rows := self.row_map[target_height][y]:
                if rows[0] < x:
                    # there is (at least) one westerly tree of `target_height`
                    # => block western visibility + update bounding box
                    # (to the eastmost westerly tree (of `target_height`))
                    vis_w = False
                    box_w = max(box_w, rows[bisect.bisect_left(rows, x) - 1])
                if rows[-1] > x:
                    vis_e = False
                    box_e = min(box_e, rows[bisect.bisect_right(rows, x)])
            if cols := self.col_map[target_height][x]:
                if cols[0] < y:
                    vis_n = False
                    box_n = max(box_n, cols[bisect.bisect_left(cols, y) - 1])
                if cols[-1] > y:
                    vis_s = False
                    box_s = min(box_s, cols[bisect.bisect_right(cols, y)])

        return TreeStats(
            vis_w or vis_e or vis_n or vis_s,
            (x - box_w) * (box_e - x) * (y - box_n) * (box_s - y),
        )

    def iter(self) -> Generator[Pos, None, None]:
        for y in range(self.size.y):
            for x in range(self.size.x):
                yield Pos(x, y)


def default_map() -> HeightMap:
    return {x: defaultdict(list) for x in range(MAX_HEIGHT + 1)}


the_map = Map.from_lines((real_data * 1).splitlines())
stats = [the_map.stats(*pos) for pos in the_map.iter()]

# Part 1 - visible trees
a1 = len([1 for _ in stats if _.visible])
print(a1)

# Part 2 - scenic score
a2 = max(_.scenic_score for _ in stats)
print(a2)
