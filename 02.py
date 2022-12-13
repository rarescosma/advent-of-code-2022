from pathlib import Path

rps = {
    "A": 1,
    "B": 2,
    "C": 3,
    "X": 1,
    "Y": 2,
    "Z": 3,
}


def play(other: int, me: int) -> int:
    if other == me:
        return 3 + me
    if win(other) == me:
        return 6 + me
    return me


def strat(o: int, s: int) -> int:
    if s == 2:
        return 3 + o
    if s == 1:
        return lose(o)
    return 6 + win(o)


def win(o: int) -> int:
    return (o % 3) + 1


def lose(o: int) -> int:
    return (o - 1) or 3


lines = Path("inputs/02.txt").read_text().splitlines()

# Part 1
p1 = sum(map(lambda x: play(rps[x[0]], rps[x[2]]), lines))
print(p1)

# Part 2
p2 = sum(map(lambda x: strat(rps[x[0]], rps[x[2]]), lines))
print(p2)
