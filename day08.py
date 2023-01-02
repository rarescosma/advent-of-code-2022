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
real: bool = True

the_data = real_data if real else test_data


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
        row_map, col_map = default_map(), default_map()

        part = cls(lines, {}, {}, Size(max_x, max_y))

        for pos in part.iter():
            tile = part[pos]
            row_map[tile][pos.y].append(pos.x)
            col_map[tile][pos.x].append(pos.y)

        return replace(part, row_map=row_map, col_map=col_map)

    def __getitem__(self, pos: Pos) -> int:
        return int(self.lines[pos.y][pos.x])

    def stats(self, pos: Pos) -> TreeStats:
        if pos == Pos(0, 0) or pos == Pos(self.size.x - 1, self.size.y - 1):
            return TreeStats(True, 0)
        _x, _y = pos

        vis_w, vis_e, vis_n, vis_s = True, True, True, True
        box_w, box_e, box_n, box_s = (
            0,
            self.size.x - 1,
            0,
            self.size.y - 1,
        )

        for target_height in range(self[pos], MAX_HEIGHT + 1):
            if rows := self.row_map[target_height][_y]:
                if rows[0] < _x:
                    # there is (at least) one westerly tree of `target_height`
                    # => block western visibility + update bounding box
                    # (to the eastmost westerly tree (of `target_height`))
                    vis_w = False
                    box_w = max(box_w, rows[bisect.bisect_left(rows, _x) - 1])
                if rows[-1] > _x:
                    vis_e = False
                    box_e = min(box_e, rows[bisect.bisect_right(rows, _x)])
            if cols := self.col_map[target_height][_x]:
                if cols[0] < _y:
                    vis_n = False
                    box_n = max(box_n, cols[bisect.bisect_left(cols, _y) - 1])
                if cols[-1] > _y:
                    vis_s = False
                    box_s = min(box_s, cols[bisect.bisect_right(cols, _y)])

        return TreeStats(
            vis_w or vis_e or vis_n or vis_s,
            (_x - box_w) * (box_e - _x) * (_y - box_n) * (box_s - _y),
        )

    def iter(self) -> Generator[Pos, None, None]:
        for _y in range(self.size.y):
            for _x in range(self.size.x):
                yield Pos(_x, _y)


def default_map() -> HeightMap:
    return {_: defaultdict(list) for _ in range(MAX_HEIGHT + 1)}


the_map = Map.from_lines((the_data * 1).splitlines())
stats = [the_map.stats(pos) for pos in the_map.iter()]

# Part 1 - visible trees
a1 = len([1 for _ in stats if _.visible])
print(a1)

# Part 2 - scenic score
a2 = max(_.scenic_score for _ in stats)
print(a2)
