import itertools
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cached_property, partial
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


Tile = int
OUT, WALL = 0, 2
Direction = int
RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3
Dim = int
X, Y, Z = 0, 1, 2
Instr = Union[int, str]
INT_RE, DIR_RE = re.compile(r"\d+"), re.compile(r"[RL]")


class Pos(NamedTuple):
    x: int
    y: int

    def __add__(self, other: tuple) -> "Pos":
        if not isinstance(other, Pos):
            raise ValueError
        return Pos(self.x + other.x, self.y + other.y)

    @classmethod
    def from_point(cls, point: "Point") -> "Pos":
        return Pos(int(point[X]), int(point[Y]))


DELTAS = [Pos(1, 0), Pos(0, 1), Pos(-1, 0), Pos(0, -1)]
MapSize = Pos
Span = tuple[int, int]
PointId = int
Edge = tuple[PointId, PointId]
Adjacency = dict[PointId, set[PointId]]


def numbers() -> Generator[PointId, None, None]:
    _i = 0
    while True:
        _i += 1
        yield _i


ID_GEN = numbers()


@dataclass(frozen=True, order=True)
class Point:
    _x: float
    _y: float
    _z: float = 0
    pid: PointId = -1

    @classmethod
    def from_pos(cls, pos: Pos) -> "Point":
        return cls(pos.x, pos.y)

    def __add__(self, other: "Point") -> "Point":
        return Point(self._x + other._x, self._y + other._y, self._z + other._z)

    def nameless(self) -> "Point":
        return replace(self, pid=-1)

    def __getitem__(self, key: Any) -> float:
        if not isinstance(key, Dim):
            raise TypeError
        if key == X:
            return self._x
        if key == Y:
            return self._y
        if key == Z:
            return self._z
        raise ValueError

    def change_dim(self, dim: Dim, func: Callable[[float], float]) -> "Point":
        update = {"_" + "xyz"[dim]: func(self[dim])}
        return replace(self, **update)

    def is_between(self, dim: Dim, a_point: "Point", b_point: "Point") -> bool:
        p_min = min(a_point[dim], b_point[dim])
        p_max = max(a_point[dim], b_point[dim])
        return p_min < self[dim] < p_max

    def forms_edge_with(self, other: "Point") -> bool:
        return (
            self._x == other[X] and abs(self._y - other[Y]) == FACE_SIZE
        ) or (self._y == other[Y] and abs(self._x - other[X]) == FACE_SIZE)

    def rotate(
        self, dim: Dim, orig: "Point", clockwise: bool = True
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


@dataclass
class Map:
    grid: dict[Pos, Tile]
    size: MapSize

    def spans(self, dim: Dim) -> list[Span]:
        _spans, _conv = [(0, 0)] * self.size[dim], converse(dim)
        for coord in range(self.size[dim]):
            line = [
                p[_conv]
                for p, tile in self.grid.items()
                if p[dim] == coord and tile != OUT
            ]
            _spans[coord] = (min(line), min(line) + len(line))
        return _spans

    @cached_property
    def row_spans(self) -> list[Span]:
        return self.spans(1)

    @cached_property
    def col_spans(self) -> list[Span]:
        return self.spans(0)

    @cached_property
    def portals(self) -> list["Portal"]:
        vertex_rows = [
            *range(0, self.size.y, FACE_SIZE),
            *range(self.size.y - 1, 0, -FACE_SIZE),
        ]
        vertices = set()
        for _y in vertex_rows:
            x_start, x_end = self.row_spans[_y]
            if _y % FACE_SIZE != 0:
                _y += 1
            for _x in range(x_start, x_end + 1, FACE_SIZE):
                vertices.add(Point(_x - 0.5, _y - 0.5, -0.5))

        geo = Geometry.from_vertices(vertices)

        return [
            portal
            for edge in geo.contour
            if (portal := Portal.from_edge(edge, self, geo)) is not None
        ]


@dataclass
class Geometry:
    vertices: set[Point]
    contour: set[Edge]
    ring: dict[PointId, PointId]
    cube_adj: Adjacency

    @classmethod
    def from_vertices(cls, vertices: set[Point]) -> "Geometry":
        # give each vertex a unique ID to track them across folds
        vertices = {replace(_, pid=next(ID_GEN)) for _ in vertices}

        squares = [
            [p0, p1, p2, p3]
            for (p0, p1, p2, p3) in itertools.permutations(vertices, r=4)
            if (p1[Y] == p0[Y] and p1[X] == p0[X] + FACE_SIZE)
            and (p2[X] == p0[X] and p2[Y] == p0[Y] + FACE_SIZE)
            and (p3[X] == p0[X] + FACE_SIZE and p3[Y] == p0[Y] + FACE_SIZE)
        ]

        common = {
            (_[0], _[1])
            for (sq0, sq1) in itertools.combinations(squares, r=2)
            if len(_ := list(set(sq0) & set(sq1))) == 2
        }

        contour = {
            (p0.pid, p1.pid)
            for (p0, p1) in itertools.combinations(vertices, r=2)
            if p0.forms_edge_with(p1) and not (common & {(p0, p1), (p1, p0)})
        }
        ring = make_ring(contour)

        folds = [Fold.from_ring(ring, (_[0].pid, _[1].pid)) for _ in common]
        cube_adj = find_cube(vertices, folds)

        return cls(vertices, contour, ring, cube_adj)


@dataclass
class Fold:
    edge: Edge
    mobile_pids: set[PointId]

    @classmethod
    def from_ring(cls, ring: dict[PointId, PointId], edge: Edge) -> "Fold":
        cur, end = edge
        mobile_pids = set()
        while ring[cur] != end:
            mobile_pids.add(ring[cur])
            cur = ring[cur]
        return cls(edge, mobile_pids)

    def fold(self, named: set[Point], clockwise: bool = True) -> set[Point]:
        get_point = partial(get_named, named)
        pt_a, pt_b = get_point(self.edge[0]), get_point(self.edge[1])
        dim = next(_ for _ in range(3) if pt_a[_] != pt_b[_])

        return {
            _.rotate(dim, pt_a, clockwise) if _.pid in self.mobile_pids else _
            for _ in named
        }


@dataclass
class Portal:
    src: tuple[Point, Point]
    dest: tuple[Point, Point]
    src_axis: Dim
    dest_axis: Dim
    dest_dir: Direction
    signum: int
    delta: Point

    @classmethod
    def from_edge(
        cls, edge: Edge, the_map: Map, geo: Geometry
    ) -> Optional["Portal"]:
        s0_pid, s1_pid = edge
        s0_adj = geo.cube_adj[s0_pid] | {s0_pid, s1_pid}
        s1_adj = geo.cube_adj[s1_pid] | {s0_pid, s1_pid}
        dest_cands = {
            (m0, m1)
            for (m0, m1) in itertools.product(s0_adj, s1_adj)
            if geo.contour & {(m0, m1), (m1, m0)}
        } - {(s0_pid, s1_pid), (s1_pid, s0_pid)}

        # not a true portal, just an inside edge
        if not dest_cands:
            return None

        get_point = partial(get_named, geo.vertices)
        dest: Edge = next(iter(dest_cands))
        dest_a, dest_b = get_point(dest[0]), get_point(dest[1])

        if dest_a[X] == dest_b[X]:
            dest_axis = X
            probe = Pos.from_point(
                Point(dest_a[X] - 0.5, (dest_a[Y] + dest_b[Y]) // 2)
            )
            if the_map.grid.get(probe, OUT) == OUT:
                dest_dir, delta = RIGHT, Point(0.5, 0)
            else:
                dest_dir, delta = LEFT, Point(-0.5, 0)
        else:
            dest_axis = Y
            probe = Pos.from_point(
                Point((dest_a[X] + dest_b[X]) // 2, dest_a[Y] - 0.5)
            )
            if the_map.grid.get(probe, OUT) == OUT:
                dest_dir, delta = DOWN, Point(0, 0.5)
            else:
                dest_dir, delta = UP, Point(0, -0.5)

        src_a, src_b = get_point(s0_pid), get_point(s1_pid)
        return cls(
            (src_a, src_b),
            (dest_a, dest_b),
            (X if src_a[X] == src_b[X] else Y),
            dest_axis,
            dest_dir,
            (-1 if (dest_a > dest_b) != (src_a > src_b) else 1),
            delta,
        )

    def passes_thru(self, the_map: "Map", old_pos: Pos, new_pos: Pos) -> bool:
        src_a, src_b = self.src
        old_pt, new_pt = Point.from_pos(old_pos), Point.from_pos(new_pos)

        dim, conv = self.src_axis, converse(self.src_axis)
        if src_a[dim] == src_b[dim] and new_pt[dim] != old_pt[dim]:
            return (
                old_pt.is_between(conv, src_a, src_b)
                and src_a.is_between(dim, old_pt, new_pt)
                and the_map.grid.get(new_pos, OUT) == OUT
            )
        return False

    def teleport(self, old_pos: Pos) -> tuple[Pos, Direction]:
        # translate from src_a to dest_a
        old_pt, offset_dim = Point.from_pos(old_pos), converse(self.src_axis)
        src_a, dest_a = self.src[0], self.dest[0]
        a_diff = old_pt[offset_dim] - src_a[offset_dim]

        new_pt = (dest_a + self.delta).change_dim(
            converse(self.dest_axis),
            lambda c: c + self.signum * a_diff,
        )
        return Pos.from_point(new_pt), self.dest_dir


def parse_map(_the_data: list[str]) -> Map:
    grid = {}
    max_y = len(_the_data)
    max_x = max(len(_) for _ in _the_data)
    for _y in range(max_y):
        row = _the_data[_y]
        for _x, tile in enumerate(row):
            grid[Pos(_x, _y)] = " .#".index(tile)
        for _x in range(len(row), max_x):
            grid[Pos(_x, _y)] = OUT
    return Map(grid, MapSize(max_x, max_y))


def parse_walk(_wline: str) -> list[Instr]:
    spans = {_.span(): int(_.group(0)) for _ in INT_RE.finditer(_wline)} | {
        _.span(): _.group(0) for _ in DIR_RE.finditer(_wline)
    }
    return [spans[k] for k in sorted(spans)]


def converse(dim: Dim) -> Dim:
    if dim == X:
        return Y
    if dim == Y:
        return X
    raise ValueError


def wrapping_move(cur: Pos, grid: Map, delta: Pos) -> Optional[Pos]:
    # get the next pos (with wrapping)
    next_pos = cur + delta

    (row_start, row_end) = grid.row_spans[cur.y]
    if next_pos.x == row_end:
        next_pos = Pos(row_start, next_pos.y)
    elif next_pos.x < row_start:
        next_pos = Pos(row_end - 1, next_pos.y)

    (col_start, col_end) = grid.col_spans[cur.x]
    if next_pos.y == col_end:
        next_pos = Pos(next_pos.x, col_start)
    elif next_pos.y < col_start:
        next_pos = Pos(next_pos.x, col_end - 1)

    if grid.grid[next_pos] == WALL:
        return None

    return next_pos


def get_named(named: set[Point], pid: PointId) -> Point:
    return next(_ for _ in named if _.pid == pid)


def make_ring(contour: set[Edge]) -> dict[PointId, PointId]:
    # make a ring
    adj: Adjacency = defaultdict(set)
    for pt_0, pt_1 in contour:
        adj[pt_0].add(pt_1)
        adj[pt_1].add(pt_0)

    cur = list(adj.keys())[0]
    seen = {cur}
    ring: dict[PointId, PointId] = {}
    while len(ring) < len(adj):
        neigh = next(iter(adj[cur]))
        ring[cur] = neigh
        adj[neigh].discard(cur)
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
                adj: Adjacency = defaultdict(set)
                for _ in _named:
                    _collate[_.nameless()].add(_.pid)
                for vertices in _collate.values():
                    for vertex in vertices:
                        adj[vertex] |= vertices - {vertex}
                return adj
    return {}


# Part 1
def part_one(m_map: Map, m_walk: list[Instr]) -> int:
    cur_pos, facing = Pos(m_map.row_spans[0][0], 0), RIGHT
    for instr in m_walk:
        if isinstance(instr, str):
            # change orientation - right is clockwise
            facing = (facing + (1 if instr == "R" else -1)) % 4
        elif isinstance(instr, int):
            # attempt walking
            walked = 0
            while walked < instr:
                new_pos = wrapping_move(cur_pos, m_map, DELTAS[facing])
                if new_pos is None:
                    break
                cur_pos = new_pos
                walked += 1
    return 1000 * (cur_pos.y + 1) + 4 * (cur_pos.x + 1) + facing


# Part 2
def part_two(m_map: Map, m_walk: list[Instr]) -> int:
    cur_pos, facing = Pos(m_map.row_spans[0][0], 0), RIGHT
    for instr in m_walk:
        if isinstance(instr, str):
            # change orientation - right is clockwise
            facing = (facing + (1 if instr == "R" else -1)) % 4
        elif isinstance(instr, int):
            # attempt walking
            walked = 0
            while walked < instr:
                new_pos = cur_pos + DELTAS[facing]
                old_facing = facing
                # teleport no matter what
                for portal in m_map.portals:
                    if portal.passes_thru(m_map, cur_pos, new_pos):
                        new_pos, facing = portal.teleport(cur_pos)
                        break
                if m_map.grid[new_pos] == WALL:
                    facing = old_facing
                    break
                cur_pos = new_pos
                walked += 1
    return 1000 * (cur_pos.y + 1) + 4 * (cur_pos.x + 1) + facing


_map, instructions = parse_map(the_data), parse_walk(the_walk)
print(part_one(_map, instructions))
print(part_two(_map, instructions))
