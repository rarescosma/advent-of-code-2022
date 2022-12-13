from collections import deque
from pathlib import Path

stack_lines = Path("inputs/05_s.txt").read_text().splitlines()

stacks = []
for line in stack_lines:
    crates = line.split(" ")
    crates.pop(0)
    stacks.append(deque(crates))


# NOTE: from_stack, to_stack are 1-indexed
def move(how_many: int, from_stack: int, to_stack: int) -> None:
    buf: deque[str] = deque([])
    for _ in range(how_many):
        buf.appendleft(stacks[from_stack - 1].pop())
    stacks[to_stack - 1].extend(buf)


move_lines = Path("inputs/05_i.txt").read_text().splitlines()

for line in move_lines:
    syms = line.split(" ")
    move(int(syms[1]), int(syms[3]), int(syms[5]))

print(stacks)
answer = [(s[-1] if s else " ") for s in stacks]

print("".join(answer))
