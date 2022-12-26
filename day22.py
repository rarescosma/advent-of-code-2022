import itertools
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Generator, NamedTuple, Optional, Union

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

real: bool = True

if real:
    the_data = [_ for _ in real_data[:-2] if _]
    the_walk = real_data[-1]
    FACE_SIZE = 50
else:
    the_data = [_ for _ in test_data[:-2] if _]
    the_walk = test_data[-1]
    FACE_SIZE = 4

INT_RE = re.compile(r"\d+")
DIR_RE = re.compile(r"[RL]")

Tile = int
OUT, EMPTY, WALL = 0, 1, 2
tiles = {
    " ": OUT,
    ".": EMPTY,
    "#": WALL,
}
Direction = int
RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3


class Pos(NamedTuple):
    x: int
    y: int

    def __add__(self, other: tuple) -> "Pos":
        if not isinstance(other, Pos):
            raise ValueError
        return Pos(self.x + other.x, self.y + other.y)

    @classmethod
    def from_point(cls, point: "Point") -> "Pos":
        return Pos(int(point.x_coord), int(point.y_coord))


deltas = [Pos(1, 0), Pos(0, 1), Pos(-1, 0), Pos(0, -1)]
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
    for _y in range(max_y):
        row = _the_data[_y]
        for _x, tile in enumerate(row):
            grid[Pos(_x, _y)] = tiles[tile]
        for _x in range(len(row), max_x):
            grid[Pos(_x, _y)] = OUT

    row_spans = [(0, 0)] * max_y
    for _y in range(max_y):
        _row = [p.x for p, tile in grid.items() if p.y == _y and tile != OUT]
        row_spans[_y] = (min(_row), min(_row) + len(_row))

    col_spans = [(0, 0)] * max_x
    for _x in range(max_x):
        _col = [p.y for p, tile in grid.items() if p.x == _x and tile != OUT]
        col_spans[_x] = (min(_col), min(_col) + len(_col))

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
    x_coord: float
    y_coord: float
    z_coord: float = 0
    pid: PointId = -1

    @classmethod
    def from_pos(cls, pos: Pos) -> "Point":
        return cls(pos.x, pos.y)

    @staticmethod
    def planar_converse(dim: int) -> int:
        if dim == 0:
            return 1
        if dim == 1:
            return 0
        raise ValueError

    def __add__(self, other: "Point") -> "Point":
        return Point(
            self.x_coord + other.x_coord,
            self.y_coord + other.y_coord,
            self.z_coord + other.z_coord,
        )

    def nameless(self) -> "Point":
        return replace(self, pid=-1)

    def __getitem__(self, key: Any) -> float:
        if not isinstance(key, int):
            raise TypeError
        if key == 0:
            return self.x_coord
        if key == 1:
            return self.y_coord
        if key == 2:
            return self.z_coord
        raise ValueError

    def change_dim(self, dim: int, func: Callable[[float], float]) -> "Point":
        update = {"xyz"[dim] + "_coord": func(self[dim])}
        return replace(self, **update)

    def is_between(self, dim: int, a_point: "Point", b_point: "Point") -> bool:
        p_min = min(a_point[dim], b_point[dim])
        p_max = max(a_point[dim], b_point[dim])
        return p_min < self[dim] < p_max

    def forms_edge_with(self, other: "Point") -> bool:
        return (
            self.x_coord == other.x_coord
            and abs(self.y_coord - other.y_coord) == FACE_SIZE
        ) or (
            self.y_coord == other.y_coord
            and abs(self.x_coord - other.x_coord) == FACE_SIZE
        )

    def rotate(
        self, dim: int, orig: "Point", clockwise: bool = True
    ) -> "Point":
        # first translate so that we go around "origin"
        coords = [self[_] - (orig[_] if _ != dim else 0) for _ in range(3)]

        l_dim = (dim - 1) % 3
        h_dim = (dim + 1) % 3
        if clockwise:
            coords[l_dim], coords[h_dim] = coords[h_dim], -coords[l_dim]
        else:
            coords[l_dim], coords[h_dim] = -coords[h_dim], coords[l_dim]

        # translate it back
        coords = [coords[_] + (orig[_] if _ != dim else 0) for _ in range(3)]
        _x, _y, _z = coords[:3]
        return Point(_x, _y, _z, pid=self.pid)


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
        pt_0, pt_1 = get_named(named, self.pid0), get_named(named, self.pid1)
        dim = next(_ for _ in range(3) if pt_0[_] != pt_1[_])

        to_rot = {
            _.rotate(dim, pt_1, clockwise)
            for _ in named
            if _.pid in self.mobile_pids
        }
        to_not = {_ for _ in named if _.pid not in self.mobile_pids}

        return to_rot | to_not


def get_named(named: set[Point], pid: PointId) -> Point:
    return next(_ for _ in named if _.pid == pid)


def make_ring(contour: set[Edge]) -> dict[PointId, PointId]:
    # make a ring
    adj: Adjacency = defaultdict(set)
    for pt_0, pt_1 in contour:
        adj[pt_0] |= {pt_1}
        adj[pt_1] |= {pt_0}

    cur = list(adj.keys())[0]
    seen = {cur}
    ring: dict[PointId, PointId] = {}
    while len(ring) < len(adj):
        neigh = next(iter(adj[cur]))
        ring[cur] = neigh
        adj[neigh] ^= {cur}
        seen.add(neigh)
        cur = neigh
    return ring


def find_cube(named: set[Point], folds: list[Fold]) -> Adjacency:
    for directions in itertools.product(*([[True, False]] * len(folds))):
        _named = deepcopy(named)
        for fold, direction in zip(folds, directions):
            _named = fold.fold(_named, direction)
            num_points = len({_.nameless() for _ in _named})
            if num_points == 8:
                _collate = defaultdict(set)
                _adj: dict[PointId, set[PointId]] = defaultdict(set)
                for _ in sorted(_named):
                    _collate[(_.x_coord, _.y_coord, _.z_coord)].add(_.pid)
                for vertices in _collate.values():
                    for vertex in vertices:
                        _adj[vertex] |= vertices - {vertex}
                return _adj
    return {}


@dataclass
class Portal:
    src: tuple[Point, Point]
    dest: tuple[Point, Point]
    src_axis: int
    dest_axis: int
    dest_dir: Direction
    signum: int
    delta: Point

    @classmethod
    def from_edge(
        cls, edge: Edge, the_map: "Map", geo: "Geometry"
    ) -> Optional["Portal"]:
        s0_pid, s1_pid = edge
        s0_adj = geo.cube_adj[s0_pid] | {s0_pid, s1_pid}
        s1_adj = geo.cube_adj[s1_pid] | {s0_pid, s1_pid}
        dest_cands = {
            (m0, m1)
            for (m0, m1) in itertools.product(s0_adj, s1_adj)
            if geo.contour & {(m0, m1), (m1, m0)}
        } - {(s0_pid, s1_pid), (s1_pid, s0_pid)}
        if not dest_cands:
            # not a true portal, just an inside edge
            return None

        dest: Edge = next(iter(dest_cands))
        get_point = partial(get_named, geo.vertices)
        d0_point, d1_point = get_point(dest[0]), get_point(dest[1])

        if d0_point.x_coord == d1_point.x_coord:
            dest_axis = 0
            probe = Pos.from_point(
                Point(
                    d0_point.x_coord - 0.5,
                    (d0_point.y_coord + d1_point.y_coord) // 2,
                )
            )
            if the_map.grid.get(probe, OUT) == OUT:
                dest_dir, delta = RIGHT, Point(0.5, 0)
            else:
                dest_dir, delta = LEFT, Point(-0.5, 0)
        else:
            dest_axis = 1
            probe = Pos.from_point(
                Point(
                    (d0_point.x_coord + d1_point.x_coord) // 2,
                    d0_point.y_coord - 0.5,
                )
            )
            if the_map.grid.get(probe, OUT) == OUT:
                dest_dir, delta = DOWN, Point(0, 0.5)
            else:
                dest_dir, delta = UP, Point(0, -0.5)

        s0_point, s1_point = get_point(s0_pid), get_point(s1_pid)
        return cls(
            (s0_point, s1_point),
            (d0_point, d1_point),
            (0 if s0_point.x_coord == s1_point.x_coord else 1),
            dest_axis,
            dest_dir,
            (-1 if (d0_point > d1_point) != (s0_point > s1_point) else 1),
            delta,
        )

    def passes_thru(self, the_map: "Map", old_pos: Pos, new_pos: Pos) -> bool:
        s0_point, s1_point = self.src
        old_pt = Point.from_pos(old_pos)
        new_pt = Point.from_pos(new_pos)

        dim = self.src_axis
        conv = Point.planar_converse(dim)
        if s0_point[dim] == s1_point[dim] and new_pt[dim] != old_pt[dim]:
            return (
                old_pt.is_between(conv, s0_point, s1_point)
                and s0_point.is_between(dim, old_pt, new_pt)
                and the_map.grid.get(new_pos, OUT) == OUT
            )
        return False

    def teleport(self, old_pos: Pos) -> tuple[Pos, Direction]:
        # translate from s0 to d0
        old_pt = Point.from_pos(old_pos)
        s0_point, _ = self.src
        d0_point, _ = self.dest
        conv = Point.planar_converse(self.src_axis)
        a_diff = old_pt[conv] - s0_point[conv]

        new_pt = (d0_point + self.delta).change_dim(
            Point.planar_converse(self.dest_axis),
            lambda c: c + self.signum * a_diff,
        )
        return Pos.from_point(new_pt), self.dest_dir


@dataclass
class Geometry:
    vertices: set[Point]
    contour: set[Edge]
    ring: dict[PointId, PointId]
    cube_adj: Adjacency

    @classmethod
    def from_vertices(cls, vertices: set[Point]) -> "Geometry":
        vertices = {replace(_, pid=next(ID_GEN)) for _ in sorted(vertices)}

        squares = [
            [p0, p1, p2, p3]
            for (p0, p1, p2, p3) in itertools.permutations(vertices, r=4)
            if (
                p1.y_coord == p0.y_coord
                and p1.x_coord == p0.x_coord + FACE_SIZE
            )
            and (
                p2.x_coord == p0.x_coord
                and p2.y_coord == p0.y_coord + FACE_SIZE
            )
            and (
                p3.x_coord == p0.x_coord + FACE_SIZE
                and p3.y_coord == p0.y_coord + FACE_SIZE
            )
        ]

        common = {
            (_[0], _[1])
            for (sq0, sq1) in itertools.combinations(squares, r=2)
            if len(_ := sorted(set(sq0) & set(sq1), key=lambda p: p.pid)) == 2
        }

        contour = {
            (p0.pid, p1.pid)
            for (p0, p1) in itertools.combinations(vertices, r=2)
            if p0.forms_edge_with(p1) and not (common & {(p0, p1), (p1, p0)})
        }
        ring = make_ring(contour)

        folds = [
            Fold.from_ring(ring, _[0].pid, _[1].pid) for _ in sorted(common)
        ]
        cube_adj = find_cube(vertices, folds)

        return cls(vertices, contour, ring, cube_adj)


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
                *range(0, self.max_y, FACE_SIZE),
                *range(self.max_y - 1, 0, -FACE_SIZE),
            ]
        )
        vxs = set()
        for _y in vertex_rows:
            x_start, x_end = self.row_spans[_y]
            if _y % FACE_SIZE != 0:
                _y += 1
            for _x in range(x_start, x_end + 1, FACE_SIZE):
                vxs.add(Point(_x - 0.5, _y - 0.5, -0.5))

        geo = Geometry.from_vertices(vxs)

        return [
            portal
            for _ in sorted(geo.contour)
            if (portal := Portal.from_edge(_, self, geo)) is not None
        ]


def wrapping_move(cur: Pos, _grid: Map, delta: Pos) -> Optional[Pos]:
    # get the next pos (with wrapping)
    next_pos = cur + delta

    (_row_start, _row_end) = _grid.row_spans[cur.y]
    if next_pos.x == _row_end:
        next_pos = Pos(_row_start, next_pos.y)
    elif next_pos.x < _row_start:
        next_pos = Pos(_row_end - 1, next_pos.y)

    (_col_start, _col_end) = _grid.col_spans[cur.x]
    if next_pos.y == _col_end:
        next_pos = Pos(next_pos.x, _col_start)
    elif next_pos.y < _col_start:
        next_pos = Pos(next_pos.x, _col_end - 1)

    if _grid.grid[next_pos] == WALL:
        return None

    return next_pos


# Part 1
m_map, m_walk = parse_map(the_data), parse_walk(the_walk)
cur_pos_p1: Pos = Pos(m_map.row_spans[0][0], 0)
FACING_P1: int = RIGHT
for instr in m_walk:
    if isinstance(instr, str):
        # change orientation - right is clockwise
        FACING_P1 = (FACING_P1 + (1 if instr == "R" else -1)) % 4
    elif isinstance(instr, int):
        # attempt walking
        walked_p1: int = 0
        while walked_p1 < instr:
            _new_pos = wrapping_move(cur_pos_p1, m_map, deltas[FACING_P1])
            if _new_pos is None:
                break
            cur_pos_p1 = _new_pos
            walked_p1 += 1

final_col, final_row = cur_pos_p1
ans1 = 1000 * (final_row + 1) + 4 * (final_col + 1) + FACING_P1
print(ans1)


# Part 2
portals = m_map.get_portals()
cur_pos_p2: Pos = Pos(m_map.row_spans[0][0], 0)
FACING_P2: int = RIGHT
for instr in m_walk:
    if isinstance(instr, str):
        # change orientation - right is clockwise
        FACING_P2 = (FACING_P2 + (1 if instr == "R" else -1)) % 4
    elif isinstance(instr, int):
        # attempt walking
        walked_p2: int = 0
        while walked_p2 < instr:
            _new_pos = cur_pos_p2 + deltas[FACING_P2]
            old_facing: int = FACING_P2
            # teleport no matter what
            for _portal in portals:
                if _portal.passes_thru(m_map, cur_pos_p2, _new_pos):
                    _new_pos, FACING_P2 = _portal.teleport(cur_pos_p2)
                    break
            if m_map.grid[_new_pos] == WALL:
                FACING_P2 = old_facing
                break
            cur_pos_p2 = _new_pos
            walked_p2 += 1

final_col, final_row = cur_pos_p2
ans2 = 1000 * (final_row + 1) + 4 * (final_col + 1) + FACING_P2
print(ans2)
