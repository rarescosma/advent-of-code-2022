from collections import defaultdict
from pathlib import Path
from textwrap import dedent

test_data = dedent(
    """
>>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>
""".strip()
)

real_data = Path("inputs/17.txt").read_text().strip()

shapes = [
    [[1, 1, 1, 1]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    [[0, 0, 1], [0, 0, 1], [1, 1, 1]],
    [[1], [1], [1], [1]],
    [[1, 1], [1, 1]],
]
dims = [(len(_[0]), len(_)) for _ in shapes]
W = 7


def find_pattern(
    max_stopped: int, find_modulus: bool = True
) -> tuple[int, ...]:
    tetris = [0] * W
    seen_states = dict()
    floor = 0
    shape_idx = 0
    wind_idx = 0
    rock_pos, rock, rock_dim = None, shapes[shape_idx], dims[shape_idx]
    stopped = 0

    the_map = defaultdict(bool)
    for _floor_x in range(W):
        the_map[(_floor_x, 0)] = True

    def collides(_rock_pos, _rock, _rock_dim) -> bool:
        if _rock_pos[0] < 0 or _rock_pos[0] + _rock_dim[0] > W:
            return True
        for _y in range(_rock_dim[1]):
            for _x in range(_rock_dim[0]):
                if not _rock[_y][_x]:
                    continue
                if the_map[(_rock_pos[0] + _x, _rock_pos[1] + _y)]:
                    return True
        return False

    def hash_state(end_floor: int, start_floor: int = 1) -> str:
        state_str = ""
        for _y in range(end_floor, start_floor):
            state_str += "".join(
                ["#" if the_map[(_x, _y)] else "." for _x in range(0, W)]
            )
            state_str += "\n"
        return state_str

    while stopped < max_stopped:
        # spawn a new rock
        if rock_pos is None:
            rock, rock_dim = shapes[shape_idx], dims[shape_idx]
            rock_pos = (2, floor - 3 - rock_dim[1])
            shape_idx = shape_idx + 1
            if shape_idx == len(shapes):
                shape_idx = 0

        wx = -1 if real_data[wind_idx] == "<" else 1
        windward = (rock_pos[0] + wx, rock_pos[1])
        if not collides(windward, rock, rock_dim):
            rock_pos = windward
        wind_idx = wind_idx + 1
        if wind_idx == len(real_data):
            wind_idx = 0

        downward = (rock_pos[0], rock_pos[1] + 1)
        if not collides(downward, rock, rock_dim):
            rock_pos = downward
        else:
            # record the rock on the map
            for y in range(rock_dim[1]):
                for x in range(rock_dim[0]):
                    occupied = bool(rock[y][x])
                    if occupied:
                        the_map[(rock_pos[0] + x, rock_pos[1] + y)] = True
                        tetris[rock_pos[0] + x] = rock_pos[1] + y
                        if rock_pos[1] < floor:
                            floor = rock_pos[1]

            if all(tetris):
                if find_modulus:
                    state_ = (
                        wind_idx,
                        shape_idx,
                        hash_state(floor, rock_pos[1] + 1),
                    )
                    if state_ not in seen_states:
                        seen_states[state_] = (stopped, floor)
                    else:
                        return (
                            stopped,
                            stopped - seen_states[state_][0],
                            seen_states[state_][1] - floor,
                        )

                # memory shenanigans
                nuke_below = max(tetris) + 5
                to_del = [k for k in the_map if k[1] > nuke_below]
                for k in to_del:
                    del the_map[k]
                tetris = [0] * W

            stopped += 1
            rock_pos = None

    return -floor, 0, 0


# Part 1
ans1 = find_pattern(2022, False)
print(ans1[0])


# Part 2
offset, modulus, height_inc = find_pattern(5 * len(real_data), True)
n_pieces = 1000000000000
repeats = (n_pieces - offset) // modulus
remains = (n_pieces - offset) % modulus
ans2 = repeats * height_inc + find_pattern(offset + remains, False)[0]
print(ans2)
