from pathlib import Path

lines = Path("inputs/01.txt").read_text().splitlines()


def solve() -> list[int]:
    elves = []
    elf = 0
    for line in lines:
        if line:
            elf += int(line)
        else:
            elves.append(elf)
            elf = 0
    return elves


# Part 1
_elves = solve()
print(max(_elves))

# Part 2
_elves.sort()
print(sum(_elves[-3:]))
