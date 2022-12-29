from copy import deepcopy
from pathlib import Path

stack_lines = Path("inputs/05_s.txt").read_text().splitlines()
move_lines = Path("inputs/05_i.txt").read_text().splitlines()

s1 = []
for line in stack_lines:
    crates = line.split(" ")
    crates.pop(0)
    s1.append(crates)

s2 = deepcopy(s1)


def move(
    stack: list[list[str]], num: int, src: int, dest: int, part: int
) -> None:
    buf, stack[src] = stack[src][-num:], stack[src][:-num]
    stack[dest].extend(buf[::-1] if part == 1 else buf)


# NOTE: src, dest are 1-indexed
for line in move_lines:
    syms = line.split(" ")
    move(s1, int(syms[1]), int(syms[3]) - 1, int(syms[5]) - 1, 1)
    move(s2, int(syms[1]), int(syms[3]) - 1, int(syms[5]) - 1, 2)


a1: str = "".join(s[-1] for s in s1)
print(a1)
a2: str = "".join(s[-1] for s in s2)
print(a2)
