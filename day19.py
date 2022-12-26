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
real: bool = True

the_data = real_data if real else test_data


CostItem = tuple[int, int]
Cost = list[CostItem]
Costs = list[Cost]
GEODE = 3


def get_costs(bp_line: str) -> list:
    return [COST_RE.findall(_) for _ in bp_line.strip().split(".")[:-1]]


def parse_blueprint(blueprint: str) -> Costs:
    costs: Costs = []
    for i, recipes in enumerate(get_costs(blueprint)):
        costs.append([])
        for recipe in recipes:
            amt, rtype = recipe.split(" ")
            costs[i].append(
                (int(amt), ["ore", "clay", "obsidian"].index(rtype))
            )
    return costs


def wait_time(costs: list[CostItem], res: list[int], bots: list[int]) -> int:
    max_wait = 0
    for amt, resource_type in costs:
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


def dfs(max_bots: list[int], costs: Costs, time: int) -> int:
    seen = set()
    queue = deque([([0, 0, 0, 0], [1, 0, 0, 0], time)])
    best = 0

    while queue:
        res, bots, time = queue.popleft()

        best = max(best, res[GEODE] + bots[GEODE] * time)

        for bot_type, blueprint in enumerate(costs):
            # Obs 1: do not build more than the max_bots number of bots
            # for each type (except geode)
            if bot_type != GEODE and bots[bot_type] >= max_bots[bot_type]:
                continue

            wait = wait_time(blueprint, res, bots)

            _t = time - wait - 1
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

            _k = tuple([time, *_res, *_bots])
            if _k not in seen:
                queue.append((_res, _bots, _t))
                seen.add(_k)

    return best


def solve(line: str, start_time: int) -> int:
    m_costs = parse_blueprint(line)
    return dfs(get_max_bots(m_costs), m_costs, start_time)


with Pool() as pool:
    # Part 1
    ans1 = sum(
        (i + 1) * v
        for i, v in enumerate(pool.map(partial(solve, start_time=24), the_data))
    )
    print(ans1)

    # Part 2
    ans2 = reduce(mul, pool.map(partial(solve, start_time=32), the_data[:3]), 1)
    print(ans2)
