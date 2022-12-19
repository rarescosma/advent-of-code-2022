from pathlib import Path
from textwrap import dedent

test_data = dedent(
    """
>>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>
""".strip()
)

W = 7
real_data = Path("inputs/17.txt").read_text().strip()
Map = set[tuple[int, int]]
Rock = set[tuple[int, int]]


def get_shape(idx: int, y: int) -> set[tuple[int, int]]:
    if idx == 0:
        return {(2, y), (3, y), (4, y), (5, y)}
    elif idx == 1:
        return {(3, y), (2, y + 1), (3, y + 1), (4, y + 1), (3, y + 2)}
    elif idx == 2:
        return {(2, y), (3, y), (4, y), (4, y + 1), (4, y + 2)}
    elif idx == 3:
        return {(2, y), (2, y + 1), (2, y + 2), (2, y + 3)}
    return {(2, y), (3, y), (2, y + 1), (3, y + 1)}


def move_rock(_rock: Rock, dx: int, dy: int) -> Rock:
    return {(_x + dx, _y + dy) for (_x, _y) in _rock}


def hash_state(_map: Map, start_y: int, end_y: int) -> frozenset:
    return frozenset(
        (_x, _y - start_y)
        for _x in range(0, W)
        for _y in range(start_y, end_y + 1)
        if (_x, _y) in _map
    )


def collides(_map: Map, _rock: Rock) -> bool:
    min_x = min(_x for (_x, _) in _rock)
    max_x = max(_x for (_x, _) in _rock)
    return min_x < 0 or max_x >= W or bool(_rock & _map)


def find_pattern(
    the_data: str, max_stopped: int, find_modulus: bool = True
) -> tuple[int, ...]:
    rock = None
    top, shape_idx, i, max_i, stopped = 0, 0, 0, len(the_data), 0

    the_map = {(_x, 0) for _x in range(0, W)}
    tetris = [0] * W
    seen = {}

    while stopped < max_stopped:
        # spawn a new rock
        if rock is None:
            rock = get_shape(shape_idx, top + 4)
            shape_idx = (shape_idx + 1) % 5

        windward = move_rock(rock, -1 if the_data[i] == "<" else 1, 0)
        if not collides(the_map, windward):
            rock = windward
        i = (i + 1) % max_i

        downward = move_rock(rock, 0, -1)
        if not downward & the_map:
            rock = downward
        else:
            # record the rock on the map
            the_map |= rock
            min_y = min(y for (_, y) in rock)
            max_y = max(y for (_, y) in rock)
            top = max(top, max_y)
            for (x, y) in rock:
                tetris[x] = y

            if find_modulus and all(tetris):
                state_ = (
                    shape_idx,
                    i,
                    hash_state(the_map, min_y, top),
                )
                if state_ not in seen:
                    seen[state_] = (stopped, top)
                else:
                    old_stopped, old_top = seen[state_]
                    return stopped, stopped - old_stopped, top - old_top
            stopped += 1
            rock = None

    return top, 0, 0


_the_data = real_data

# Part 1
ans1 = find_pattern(_the_data, 2022, False)
print(ans1[0])


# Part 2
n_pieces = 1000000000000
offset, modulus, _dy = find_pattern(_the_data, 5 * len(_the_data), True)
repeats = (n_pieces - offset) // modulus
remains = (n_pieces - offset) % modulus
ans2 = repeats * _dy + find_pattern(_the_data, offset + remains, False)[0]
print(ans2)
