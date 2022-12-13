from pathlib import Path

rps = {
    "A": 1,
    "B": 2,
    "C": 3,
    "X": 1,
    "Y": 2,
    "Z": 3,
}


def play(other, me):
    if other == me:
        return 3 + me
    if win(other) == me:
        return 6 + me
    return me


def strat(o, s):
    if s == 2:
        return 3 + o
    if s == 1:
        return lose(o)
    if s == 3:
        return 6 + win(o)


def win(o):
    return (o % 3) + 1


def lose(o):
    return (o - 1) or 3


with Path("inputs/02.txt") as f:
    lines = f.read_text().splitlines()

    rounds = map(lambda x: play(rps[x[0]], rps[x[2]]), lines)
    print(sum(rounds))
