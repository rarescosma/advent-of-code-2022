from pathlib import Path

lines = Path("inputs/03.txt").read_text().splitlines()


def to_prio(letter: str) -> int:
    big = ord("A")
    smol = ord("a")
    if ord(letter) >= smol:
        return ord(letter) - smol + 1
    return ord(letter) - big + 27


def to_prios(letters: str) -> list[int]:
    return [to_prio(letter) for letter in letters]


# Part 1
p1 = sum(
    map(
        lambda l: (
            set(to_prios(l[len(l) // 2 :])) & set(to_prios(l[: len(l) // 2]))
        ).pop(),
        lines,
    )
)
print(p1)

# Part 2
p2 = 0
for x in range(0, len(lines), 3):
    l1 = set(to_prios(lines[x]))
    l2 = set(to_prios(lines[x + 1]))
    l3 = set(to_prios(lines[x + 2]))
    p2 += (l1 & l2 & l3).pop()
print(p2)
