from pathlib import Path

lines = Path("inputs/03.txt").read_text().splitlines()


def to_prio(letter: str) -> int:
    big = ord("A")
    smol = ord("a")
    if ord(letter) >= smol:
        return ord(letter) - smol + 1
    return ord(letter) - big + 27


def to_prios(letters: str) -> set[int]:
    return {to_prio(letter) for letter in letters}


# Part 1
p1 = sum(
    (to_prios(line[len(line) // 2 :]) & to_prios(line[: len(line) // 2])).pop()
    for line in lines
)
print(p1)

# Part 2
p2: int = 0
for x in range(0, len(lines), 3):
    l1 = to_prios(lines[x])
    l2 = to_prios(lines[x + 1])
    l3 = to_prios(lines[x + 2])
    p2 += (l1 & l2 & l3).pop()
print(p2)
