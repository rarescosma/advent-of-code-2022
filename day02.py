from pathlib import Path

rps = {
    "A": 1,
    "B": 2,
    "C": 3,
    "X": 1,
    "Y": 2,
    "Z": 3,
}


def play(other: int, myself: int) -> int:
    if other == myself:
        return 3 + myself
    if win(other) == myself:
        return 6 + myself
    return myself


def strat(other: int, myself: int) -> int:
    if myself == 2:
        return 3 + other
    if myself == 1:
        return lose(other)
    return 6 + win(other)


def win(other: int) -> int:
    return (other % 3) + 1


def lose(other: int) -> int:
    return (other - 1) or 3


lines = Path("inputs/02.txt").read_text().splitlines()

# Part 1
p1 = sum(map(lambda x: play(rps[x[0]], rps[x[2]]), lines))
print(p1)

# Part 2
p2 = sum(map(lambda x: strat(rps[x[0]], rps[x[2]]), lines))
print(p2)
