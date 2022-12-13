from pathlib import Path

lines = Path("inputs/01.txt").read_text().splitlines()

p1 = 0
elves = []
elf = 0
for line in lines:
    if not line:
        elves.append(elf)
        elf = 0
    else:
        elf += int(line)
    p1 = max(p1, elf)

# Part 1
print(p1)

# Part 2
elves.sort(reverse=True)
print(sum(elves[:3]))
