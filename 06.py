from pathlib import Path

real_data = Path("inputs/06.txt").read_text()


def distinct_index(_input: str, how_many: int) -> int:
    for i in range(len(_input)):
        window = _input[i : i + how_many]
        if len(window) == len(set(window)):
            return i + how_many
    return -1


# Part 1
a1 = distinct_index(real_data, 4)
print(a1)

# Part 2
a2 = distinct_index(real_data, 14)
print(a2)
