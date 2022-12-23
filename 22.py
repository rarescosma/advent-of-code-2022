import itertools
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from textwrap import dedent
from typing import Generator, Optional, Union

test_data = dedent(
    """
        ...#
        .#..
        #...
        ....
...#.......#
........#...
..#....#....
..........#.
        ...#....
        .....#..
        .#......
        ......#.

10R5L5R10L4R5L5
"""
).splitlines()

real_data = Path("inputs/22.txt").read_text().splitlines()

real = True

if real:
    the_data = [_ for _ in real_data[:-2] if _]
    the_walk = real_data[-1]
    face_size = 50
else:
    the_data = [_ for _ in test_data[:-2] if _]
    the_walk = test_data[-1]
    face_size = 4

INT_RE = re.compile(r"\d+")
DIR_RE = re.compile(r"[RL]")

Tile = int
OUT, EMPTY, WALL = 0, 1, 2
tiles = {
    " ": OUT,
    ".": EMPTY,
    "#": WALL,
}
b_tiles = [" ", ".", "#"]
Direction = int
RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3
deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]


Pos = tuple[int, int]
Span = tuple[int, int]
Instr = Union[int, str]


def parse_walk(_wline: str) -> list[Instr]:
    spans = {_.span(): int(_.group(0)) for _ in INT_RE.finditer(_wline)} | {
        _.span(): _.group(0) for _ in DIR_RE.finditer(_wline)
    }
    return [spans[k] for k in sorted(spans)]


def parse_map(_the_data: list[str]) -> "Map":
    grid = {}
    max_y = len(_the_data)
    max_x = max(len(_) for _ in _the_data)
    for y in range(max_y):
        row = _the_data[y]
        for x, tile in enumerate(row):
            grid[(x, y)] = tiles[tile]
        for x in range(len(row), max_x):
            grid[(x, y)] = OUT

    row_spans = [(0, 0)] * max_y
    for y in range(max_y):
        _row = [_x for (_x, _y) in grid if _y == y and grid[(_x, _y)] != OUT]
        row_spans[y] = (min(_row), min(_row) + len(_row))

    col_spans = [(0, 0)] * max_x
    for x in range(max_x):
        _col = [_y for (_x, _y) in grid if _x == x and grid[(_x, _y)] != OUT]
        col_spans[x] = (min(_col), min(_col) + len(_col))

    return Map(grid, max_x, max_y, row_spans, col_spans)


def numbers() -> Generator[int, None, None]:
    _i = 0
    while True:
        _i += 1
        yield _i


ID_GEN = numbers()
PointId = int


@dataclass(frozen=True, order=True)
class Point:
    x: float
    y: float
    z: float
    pid: PointId = -1

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def nameless(self) -> "Point":
        return replace(self, pid=-1)

    def dim(self, n: int) -> float:
        if n < 0 or n >= 3:
            raise ValueError
        return [self.x, self.y, self.z][n]

    def forms_edge_with(self, p: "Point") -> bool:
        return (self.x == p.x and abs(self.y - p.y) == face_size) or (
            self.y == p.y and abs(self.x - p.x) == face_size
        )

    def rotate(
        self, dim: int, orig: "Point", clockwise: bool = True
    ) -> "Point":
        # first translate so that we go around "origin"
        coords = [
            self.dim(_) - (orig.dim(_) if _ != dim else 0) for _ in range(3)
        ]

        l_dim = (dim - 1) % 3
        h_dim = (dim + 1) % 3
        if clockwise:
            coords[l_dim], coords[h_dim] = coords[h_dim], -coords[l_dim]
        else:
            coords[l_dim], coords[h_dim] = -coords[h_dim], coords[l_dim]

        # translate it back
        coords = [
            coords[_] + (orig.dim(_) if _ != dim else 0) for _ in range(3)
        ]
        x, y, z = coords[:3]
        return Point(x, y, z, pid=self.pid)


Edge = tuple[PointId, PointId]
Adjacency = dict[PointId, set[PointId]]


@dataclass
class Fold:
    pid0: PointId
    pid1: PointId
    mobile_pids: set[PointId]

    @classmethod
    def from_ring(
        cls, ring: dict[PointId, PointId], pid0: PointId, pid1: PointId
    ) -> "Fold":
        cur = pid0
        mobile_pids = set()
        while ring[cur] != pid1:
            mobile_pids.add(ring[cur])
            cur = ring[cur]
        return cls(pid0, pid1, mobile_pids)

    def fold(self, named: set[Point], clockwise: bool = True) -> set[Point]:
        p0, p1 = get_named(named, self.pid0), get_named(named, self.pid1)
        dim = next(_ for _ in range(3) if p0.dim(_) != p1.dim(_))

        to_rot = {
            _.rotate(dim, p1, clockwise)
            for _ in named
            if _.pid in self.mobile_pids
        }
        to_not = {_ for _ in named if _.pid not in self.mobile_pids}

        return to_rot | to_not


def get_named(named: set[Point], pid: PointId) -> Point:
    return next(_ for _ in named if _.pid == pid)


def find_cube(named: set[Point], folds: list[Fold]) -> Adjacency:
    for directions in itertools.product(*([[True, False]] * len(folds))):
        _named = deepcopy(named)
        for fold, direction in zip(folds, directions):
            _named = fold.fold(_named, direction)
            _nameless = {_.nameless() for _ in _named}
            num_points = len(_nameless)
            if num_points == 8:
                _collate = defaultdict(set)
                _adj: dict[PointId, set[PointId]] = defaultdict(set)
                for p in sorted(_named):
                    _collate[(p.x, p.y, p.z)].add(p.pid)
                for vxs in _collate.values():
                    for vx in vxs:
                        _adj[vx] |= vxs - {vx}
                return _adj
    return {}


@dataclass
class Portal:
    src: tuple[Point, Point]
    dest: tuple[Point, Point]
    dest_dir: Direction
    signum: int
    delta: Point

    @classmethod
    def from_edge(
        cls,
        edge: Edge,
        all_edges: set[Edge],
        adj: Adjacency,
        the_map: "Map",
        vxs: set[Point],
    ) -> Optional["Portal"]:
        s0, s1 = edge
        s0_adj = adj[s0] | {s0, s1}
        s1_adj = adj[s1] | {s0, s1}
        dest_cands = {
            (m0, m1)
            for (m0, m1) in itertools.product(s0_adj, s1_adj)
            if all_edges & {(m0, m1), (m1, m0)}
        }
        if not dest_cands - {(s0, s1), (s1, s0)}:
            # not a true portal, just an inside edge
            return None

        dest: Edge = next(iter(dest_cands - {(s0, s1), (s1, s0)}))
        d0, d1 = get_named(vxs, dest[0]), get_named(vxs, dest[1])
        if d0.x == d1.x:
            probe_y = int((d0.y + d1.y) // 2)
            if the_map.grid.get((int(d0.x - 0.5), probe_y), OUT) == OUT:
                dest_dir = RIGHT
                delta = Point(0.5, 0, 0)
            else:
                dest_dir = LEFT
                delta = Point(-0.5, 0, 0)
        else:
            probe_x = int((d0.x + d1.x) // 2)
            if the_map.grid.get((probe_x, int(d0.y - 0.5)), OUT) == OUT:
                dest_dir = DOWN
                delta = Point(0, 0.5, 0)
            else:
                dest_dir = UP
                delta = Point(0, -0.5, 0)

        s0_point, s1_point = get_named(vxs, s0), get_named(vxs, s1)
        return cls(
            (s0_point, s1_point),
            (d0, d1),
            dest_dir,
            (-1 if (d0 > d1) != (s0_point > s1_point) else 1),
            delta,
        )

    def passes_thru(self, the_map: "Map", old_pos: Pos, new_pos: Pos) -> bool:
        opx, opy = old_pos
        npx, npy = new_pos
        s0, s1 = self.src

        if s0.x == s1.x and npx != opx:
            y_min, y_max = min(s0.y, s1.y), max(s0.y, s1.y)
            return (
                y_min < opy < y_max
                and (npx < s0.x < opx or opx < s0.x < npx)
                and the_map.grid.get(new_pos, OUT) == OUT
            )

        if s0.y == s1.y and npy != opy:
            x_min, x_max = min(s0.x, s1.x), max(s0.x, s1.x)
            return (
                x_min < opx < x_max
                and (npy < s0.y < opy or opy < s0.y < npy)
                and the_map.grid.get(new_pos, OUT) == OUT
            )

        return False

    def teleport(self, old_pos: Pos) -> tuple[Pos, Direction]:
        # translate from s0 to d0
        s0, s1 = self.src
        d0, d1 = self.dest
        opx, opy = old_pos
        if s0.x == s1.x:
            a_diff = opy - s0.y
        else:
            a_diff = opx - s0.x

        new_pos = d0 + self.delta
        if d0.x == d1.x:
            new_y = new_pos.y + self.signum * a_diff
            return (int(new_pos.x), int(new_y)), self.dest_dir
        else:
            new_x = new_pos.x + self.signum * a_diff
            return (int(new_x), int(new_pos.y)), self.dest_dir


@dataclass
class Map:
    grid: dict[Pos, Tile]
    max_x: int
    max_y: int
    row_spans: list[Span]
    col_spans: list[Span]

    def get_portals(self) -> list[Portal]:
        vertex_rows = sorted(
            [
                *range(0, self.max_y, face_size),
                *range(self.max_y - 1, 0, -face_size),
            ]
        )
        vxs = set()
        for y in vertex_rows:
            x0, x1 = self.row_spans[y]
            if y % face_size != 0:
                y += 1
            for x in range(x0, x1 + 1, face_size):
                vxs.add(Point(x - 0.5, y - 0.5, -0.5))

        vxs = {replace(_, pid=next(ID_GEN)) for _ in sorted(vxs)}

        # hmm, perhaps we should just consider edges instead
        # and fold on edges that don't have a barrier with the outside
        squares = [
            [p0, p1, p2, p3]
            for (p0, p1, p2, p3) in itertools.permutations(vxs, r=4)
            if p1.y == p0.y
            and p1.x == p0.x + face_size
            and p2.x == p0.x
            and p2.y == p0.y + face_size
            and p3.x == p0.x + face_size
            and p3.y == p0.y + face_size
        ]

        common = {
            (edge[0], edge[1])
            for (sq0, sq1) in itertools.combinations(squares, r=2)
            if len(
                edge := sorted(list(set(sq0) & set(sq1)), key=lambda p: p.pid)
            )
            == 2
        }

        edges = {
            (p0.pid, p1.pid)
            for (p0, p1) in itertools.combinations(vxs, r=2)
            if p0.forms_edge_with(p1) and not (common & {(p0, p1), (p1, p0)})
        }

        # make a ring
        adj: Adjacency = defaultdict(set)
        for p0, p1 in edges:
            adj[p0] |= {p1}
            adj[p1] |= {p0}

        cur = list(adj.keys())[0]
        seen = {cur}
        ring: dict[PointId, PointId] = {}
        while len(ring) < len(adj):
            neigh = next(iter(adj[cur]))
            ring[cur] = neigh
            adj[neigh] ^= {cur}
            seen.add(neigh)
            cur = neigh

        folds = [
            Fold.from_ring(ring, _[0].pid, _[1].pid)
            for _ in sorted(common)
            if len(_) == 2
        ]
        adj = find_cube(vxs, folds)

        return [
            portal
            for _ in sorted(edges)
            if (portal := Portal.from_edge(_, edges, adj, self, vxs))
            is not None
        ]


def wrapping_move(cur: Pos, _grid: Map, delta: Pos) -> Optional[Pos]:
    # get the next pos (with wrapping)
    x, y = cur
    dx, dy = delta
    nx, ny = (x + dx, y + dy)

    (_row_start, _row_end) = _grid.row_spans[y]
    if nx == _row_end:
        nx = _row_start
    elif nx < _row_start:
        nx = _row_end - 1

    (_col_start, _col_end) = _grid.col_spans[x]
    if ny == _col_end:
        ny = _col_start
    elif ny < _col_start:
        ny = _col_end - 1

    if _grid.grid[(nx, ny)] == WALL:
        return None

    return nx, ny


m_map, m_walk = parse_map(the_data), parse_walk(the_walk)
cur_pos = m_map.row_spans[0][0], 0
facing = RIGHT
for instr in m_walk:
    if isinstance(instr, str):
        # change orientation - right is clockwise
        facing = (facing + (1 if instr == "R" else -1)) % 4
    elif isinstance(instr, int):
        # attempt walking
        walked = 0
        while walked < instr:
            _new_pos = wrapping_move(cur_pos, m_map, deltas[facing])
            if _new_pos is None:
                break
            cur_pos = _new_pos
            walked += 1

# Part 1
final_col, final_row = cur_pos
ans1 = 1000 * (final_row + 1) + 4 * (final_col + 1) + facing
print(ans1)

# Part 2
portals = m_map.get_portals()
cur_pos = m_map.row_spans[0][0], 0
facing = RIGHT
for instr in m_walk:
    if isinstance(instr, str):
        # change orientation - right is clockwise
        facing = (facing + (1 if instr == "R" else -1)) % 4
    elif isinstance(instr, int):
        # attempt walking
        walked = 0
        while walked < instr:
            _new_pos = (
                cur_pos[0] + deltas[facing][0],
                cur_pos[1] + deltas[facing][1],
            )
            old_facing = facing
            # teleport no matter what
            for _portal in portals:
                if _portal.passes_thru(m_map, cur_pos, _new_pos):
                    _new_pos, facing = _portal.teleport(cur_pos)
                    break
            if m_map.grid[_new_pos] == WALL:
                facing = old_facing
                break
            cur_pos = _new_pos
            walked += 1

final_col, final_row = cur_pos
ans2 = 1000 * (final_row + 1) + 4 * (final_col + 1) + facing
print(ans2)
