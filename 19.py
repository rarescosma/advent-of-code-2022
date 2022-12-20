import math
import re
from collections import deque
from functools import partial, reduce
from multiprocessing import Pool
from operator import mul
from pathlib import Path
from typing import Pattern

COST_RE: Pattern = re.compile(r"\d+\s[a-z]+")

test_data = Path("inputs/19_t.txt").read_text().splitlines()
real_data = Path("inputs/19.txt").read_text().splitlines()

CostItem = tuple[int, int]
Cost = list[CostItem]
Costs = list[Cost]
GEODE = 3


def get_costs(bp_line: str) -> list:
    return [COST_RE.findall(_) for _ in bp_line.strip().split(".")[:-1]]


def parse_blueprint(bp: str) -> Costs:
    blueprint: Costs = []
    for i, recipes in enumerate(get_costs(bp)):
        blueprint.append([])
        for recipe in recipes:
            amt, rtype = recipe.split(" ")
            blueprint[i].append(
                (int(amt), ["ore", "clay", "obsidian"].index(rtype))
            )
    return blueprint


def wait_time(c: list[CostItem], res: list[int], bots: list[int]) -> int:
    max_wait = 0
    for amt, resource_type in c:
        if bots[resource_type] == 0:
            return 2**8
        max_wait = max(
            max_wait,
            math.ceil((amt - res[resource_type]) / bots[resource_type]),
        )
    return max_wait


def get_max_bots(costs: Costs) -> list[int]:
    max_bots = [0, 0, 0]
    for blueprint in costs:
        for (amt, rtype) in blueprint:
            max_bots[rtype] = max(max_bots[rtype], amt)
    return max_bots


def dfs(max_bots: list[int], costs: Costs, t: int) -> int:
    seen = set()
    q = deque([([0, 0, 0, 0], [1, 0, 0, 0], t)])
    best = 0

    while q:
        res, bots, t = q.popleft()

        best = max(best, res[GEODE] + bots[GEODE] * t)

        for bot_type, blueprint in enumerate(costs):
            # Obs 1: do not build more than the max_bots number of bots
            # for each type (except geode)
            if bot_type != GEODE and bots[bot_type] >= max_bots[bot_type]:
                continue

            wait = wait_time(blueprint, res, bots)

            _t = t - wait - 1
            if _t <= 0:
                continue
            _bots = bots[:]
            _res = [old + rate * (wait + 1) for old, rate in zip(res, bots)]
            _bots[bot_type] += 1
            for amt, res_type in blueprint:
                _res[res_type] -= amt

            # Obs 2: throw away extra resources
            for res_type in range(3):
                _res[res_type] = min(_res[res_type], max_bots[res_type] * _t)

            _k = tuple([t, *_res, *_bots])
            if _k not in seen:
                q.append((_res, _bots, _t))
                seen.add(_k)

    return best


def solve(line: str, t: int) -> int:
    m_costs = parse_blueprint(line)
    return dfs(get_max_bots(m_costs), m_costs, t)


with Pool() as pool:
    # Part 1
    ans1 = sum(
        (i + 1) * v
        for i, v in enumerate(pool.map(partial(solve, t=24), real_data))
    )
    print(ans1)

    # Part 2
    ans2 = reduce(mul, pool.map(partial(solve, t=32), real_data[:3]), 1)
    print(ans2)
