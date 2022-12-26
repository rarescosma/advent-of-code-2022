from pathlib import Path
from textwrap import dedent

test_data = dedent(
    """
>>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>
""".strip()
)

W = 7
HASH_WINDOW = 10
real_data = Path("inputs/17.txt").read_text().strip()
Solid = set[complex]
Rock = set[complex]
shapes: list[Rock] = [
    {0, 1, 2, 3},
    {1, 1j, 1 + 1j, 2 + 1j, 1 + 2j},
    {0, 1, 2, 2 + 1j, 2 + 2j},
    {0, 1j, 2j, 3j},
    {0, 1, 1j, 1 + 1j},
]


def move_rock(_rock: Rock, dc: complex) -> Rock:
    return {_ + dc for _ in _rock}


def hash_state(_map: Solid, end_y: int) -> frozenset:
    start_y = end_y - HASH_WINDOW
    return frozenset(
        complex(_x + (_y - start_y))
        for _x in range(0, W)
        for _y in range(start_y, end_y + 1)
        if complex(_x, _y) in _map
    )


def find_pattern(
    jets: list[int], max_stopped: int, find_cycle: bool = True
) -> tuple[int, ...]:
    rock = None
    top, shape_idx, i, max_i, stopped = 0, 0, 0, len(jets), 0

    solid: set[complex] = set(range(0, W))
    seen = {}

    while stopped < max_stopped:
        # spawn a new rock
        if rock is None:
            rock = {_ + 2 + (top + 4) * 1j for _ in shapes[shape_idx]}
            shape_idx = (shape_idx + 1) % 5

        moved = move_rock(rock, jets[i])
        if all(0 <= r.real < W for r in moved) and not moved & solid:
            rock = moved
        i = (i + 1) % max_i

        moved = move_rock(rock, -1j)
        if not moved & solid:
            rock = moved
            continue

        # record the rock on the map + push top up
        solid |= rock
        top = max(top, int(max(_.imag for _ in rock)))

        if find_cycle:
            state_ = (shape_idx, i, hash_state(solid, top))
            if state_ not in seen:
                seen[state_] = (stopped, top)
            else:
                old_stopped, old_top = seen[state_]
                return stopped, stopped - old_stopped, top - old_top

        stopped += 1
        rock = None

    return top, 0, 0


_jets = [-1 if c == "<" else 1 for c in real_data]

# Part 1
ans1 = find_pattern(_jets, 2022, False)
print(ans1[0])


# Part 2
n_pieces = 1_000_000_000_000
offset, modulus, _dy = find_pattern(_jets, 5 * len(_jets), True)
quot, rem = divmod(n_pieces - offset, modulus)
ans2 = quot * _dy + find_pattern(_jets, offset + rem, False)[0]
print(ans2)
