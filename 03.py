from pathlib import Path

lines = Path("inputs/03.txt").read_text().splitlines()


def to_prio(l):
    big = ord("A")
    smol = ord("a")
    if ord(l) >= smol:
        return ord(l) - smol + 1
    return ord(l) - big + 27


def to_prios(ls):
    return [to_prio(l) for l in ls]


cp = map(
    lambda l: set(to_prios(l[len(l) // 2 :])) & set(to_prios(l[: len(l) // 2])),
    lines,
)

print(sum(map(lambda s: s.pop(), cp)))

acc = 0
for x in range(0, len(lines), 3):
    l1 = set(to_prios(lines[x]))
    l2 = set(to_prios(lines[x + 1]))
    l3 = set(to_prios(lines[x + 2]))
    badge = (l1 & l2 & l3).pop()
    acc = acc + badge

print(acc)
