from collections import deque
from copy import deepcopy
from pathlib import Path

stack_lines = Path("inputs/05_s.txt").read_text().splitlines()

s1 = []
for line in stack_lines:
    crates = line.split(" ")
    crates.pop(0)
    s1.append(deque(crates))

s2 = deepcopy(s1)


# NOTE: from_stack, to_stack are 1-indexed
def move(
    stack: list, how_many: int, from_stack: int, to_stack: int, part: int
) -> None:
    buf: deque[str] = deque([])
    for _ in range(how_many):
        if part == 1:
            buf.append(stack[from_stack - 1].pop())
        else:
            buf.appendleft(stack[from_stack - 1].pop())
    stack[to_stack - 1].extend(buf)


move_lines = Path("inputs/05_i.txt").read_text().splitlines()

for line in move_lines:
    syms = line.split(" ")
    move(s1, int(syms[1]), int(syms[3]), int(syms[5]), 1)
    move(s2, int(syms[1]), int(syms[3]), int(syms[5]), 2)


a1: str = "".join((s[-1] if s else " ") for s in s1)
print(a1)
a2: str = "".join((s[-1] if s else " ") for s in s2)
print(a2)
