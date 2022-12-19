import re
from collections import deque
from pathlib import Path
from typing import Generator, Sequence

VALVE_RE = re.compile("[A-Z]+")
INT_RE = re.compile(r"\d+")

# Graph is bidirectional!
real_data = Path("inputs/16_t.txt").read_text()


def all_nums() -> Generator[int, None, None]:
    x = 0
    while True:
        yield x
        x += 1


NUMS = all_nums()
valve_nums = {}


def valve_to_num(v: str) -> int:
    if v not in valve_nums:
        valve_nums[v] = next(NUMS)
    return valve_nums[v]


def extract_int(v: str) -> int:
    return int(next(iter(INT_RE.findall(v))))


lines = sorted(real_data.splitlines(), key=extract_int, reverse=True)
num_valves = len(lines)
valves: list[list[int]] = [[] for _ in range(num_valves)]
flows = [0 for _ in range(num_valves)]

for line in real_data.splitlines():
    from_v, *to_v = map(valve_to_num, VALVE_RE.findall(line[1:]))
    valves[from_v] = to_v
    if (flow := extract_int(line)) != 0:
        flows[from_v] = flow

aa_valve = valve_nums["AA"]


def compute_distances(start: int) -> list[int]:
    # minimum number of steps to reach all other nodes starting at start
    seen = {start}
    q = deque(valves[start])
    depth = 0

    ret: list[int] = [0 for _ in range(num_valves)]
    q_a: list[int] = []
    while len(seen) < num_valves:
        q.extend(q_a)
        depth += 1
        q_a.clear()
        while q:
            orig = q.popleft()
            seen.add(orig)
            ret[orig] = depth
            q_a.extend([_ for _ in valves[orig] if _ not in seen])

    return ret


distances = [compute_distances(_) for _ in range(num_valves)]


def rest_key(rest: Sequence[int]) -> int:
    k = 0
    for r in rest:
        k += 1 << r
    return k


def choose_one(
    cands: int,
) -> Generator[tuple[int, int], None, None]:
    for i in range(num_valves):
        if (1 << i) & cands:
            yield i, (cands ^ (1 << i))


_flows = tuple(i for i, f in enumerate(flows) if f)
len_flows = len(_flows)
closed_valves = rest_key(_flows)
DP: dict[int, int] = {}


def dfs(cur: int, rest: int, t: int, part2: bool = False) -> int:
    if t <= 0 or not rest:
        return 0

    d_key = rest * len_flows * 31 * 2 + cur * 31 * 2 + t * 2 + int(part2)
    if (cached := DP.get(d_key)) is not None:
        return cached

    scores = [dfs(aa_valve, rest, 26)] if part2 else []

    for _cand, _rest in choose_one(rest):
        if (_dist := distances[cur][_cand]) >= t:
            continue
        score = flows[_cand] * (t - _dist - 1) + dfs(
            _cand,
            _rest,
            t - _dist - 1,
            part2,
        )
        scores.append(score)

    ret = 0 if not scores else max(scores)
    DP[d_key] = ret
    return ret


# Part 1
ans = dfs(aa_valve, closed_valves, 30)
print(ans)

# Part 2
ans2 = dfs(aa_valve, closed_valves, 26, part2=True)
print(ans2)
