from pathlib import Path

lines = Path("inputs/01.txt").read_text().splitlines()

elves = []
elf = 0
for line in lines:
    if line:
        elf += int(line)
    else:
        elves.append(elf)
        elf = 0

# Part 1
print(max(elves))

# Part 2
elves.sort()
print(sum(elves[-3:]))
