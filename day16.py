import re
from collections import deque
from pathlib import Path

VALVE_RE = re.compile("[A-Z]+")
INT_RE = re.compile(r"\d+")

# Graph is bidirectional!
real_data = Path("inputs/16.txt").read_text()


def extract_int(from_str: str) -> int:
    return int(next(iter(INT_RE.findall(from_str))))


lines = sorted(real_data.splitlines(), key=extract_int, reverse=True)
num_valves = len(lines)
valves = {}
flows = {}

for line in real_data.splitlines():
    from_v, *to_v = VALVE_RE.findall(line[1:])
    valves[from_v] = set(to_v)
    if (flow := extract_int(line)) != 0:
        flows[from_v] = flow


def compute_distances(start: str) -> dict[str, int]:
    # minimum number of steps to reach all other nodes starting at start
    seen = {start}
    queue = deque(valves[start])
    depth = 0

    ret = {}
    q_a: list[str] = []
    while len(seen) < num_valves:
        queue.extend(q_a)
        depth += 1
        q_a.clear()
        while queue:
            orig = queue.popleft()
            seen.add(orig)
            ret[orig] = depth
            q_a.extend(valves[orig] - seen)
    return ret


distances = {"AA": compute_distances("AA")}
indices = {}
for i, valve in enumerate(flows):
    distances[valve] = compute_distances(valve)
    indices[valve] = i


cache: dict[tuple, int] = {}


def dfs(cur: str, bitmask: int, time: int, part2: bool = False) -> int:
    if (cur, bitmask, time, part2) in cache:
        return cache[(cur, bitmask, time, part2)]

    score = dfs("AA", bitmask, 26) if part2 else 0

    for neighbor in distances[cur]:
        if neighbor not in flows:
            continue
        bit = 1 << indices[neighbor]
        if bitmask & bit:
            continue
        rem_t = time - distances[cur][neighbor] - 1
        if rem_t <= 0:
            continue
        score = max(
            score,
            dfs(neighbor, bitmask | bit, rem_t, part2)
            + flows[neighbor] * rem_t,
        )

    cache[(cur, bitmask, time, part2)] = score
    return score


# Part 1
ans = dfs("AA", 0, 30)
print(ans)

# Part 2
ans2 = dfs("AA", 0, 26, part2=True)
print(ans2)
