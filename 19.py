import math
import re
from pathlib import Path
from typing import Pattern

COST_RE: Pattern = re.compile(r"\d+\s[a-z]+")

test_data = Path("inputs/19_t.txt").read_text().splitlines()
real_data = Path("inputs/19.txt").read_text().splitlines()

CostItem = tuple[int, int]
Cost = list[CostItem]
Costs = list[Cost]


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


def time_before_build(
    c: tuple[CostItem, ...], res: tuple[int, ...], rob: tuple[int, ...]
) -> int:
    max_wait = 0
    for amt, resource_type in c:
        if rob[resource_type] == 0:
            return 2**8
        max_wait = max(
            max_wait,
            math.ceil((amt - res[resource_type]) / rob[resource_type]),
        )
    return max_wait


def get_max_robs(costs: Costs) -> list[int]:
    max_robs = [0, 0, 0]
    for blueprint in costs:
        for (amt, rtype) in blueprint:
            max_robs[rtype] = max(max_robs[rtype], amt)
    return max_robs


def dfs(
    max_robs: list[int],
    costs: Costs,
    cache: dict,
    res: list[int],
    rob: list[int],
    t: int,
) -> int:
    key = tuple([t, *res, *rob])
    if key in cache:
        return cache[key]

    maxval = res[3] + rob[3] * t

    if t == 0:
        return maxval

    for robot_type, blueprint in enumerate(costs):
        # do not build more than the max_robs number of robots
        # for each type (except geode, which is 3)
        if robot_type != 3 and rob[robot_type] >= max_robs[robot_type]:
            continue

        wait = time_before_build(tuple(blueprint), tuple(res), tuple(rob))

        _t = t - wait - 1
        if _t <= 0:
            continue
        _rob = list(rob)
        _res = [old + rate * (wait + 1) for old, rate in zip(res, rob)]
        for amt, resource_type in blueprint:
            _res[resource_type] -= amt
        _rob[robot_type] += 1

        # Throw away extra resources
        for resource_type in range(3):
            _res[resource_type] = min(
                _res[resource_type], max_robs[resource_type] * _t
            )

        maxval = max(maxval, dfs(max_robs, costs, cache, _res, _rob, _t))

    cache[key] = maxval
    return maxval


# Part 1
ans1 = 0
for _i, line in enumerate(real_data):
    m_costs = parse_blueprint(line)
    m_max = get_max_robs(m_costs)
    max_val = dfs(m_max, m_costs, {}, [0, 0, 0, 0], [1, 0, 0, 0], 24)
    ans1 += (_i + 1) * max_val
print(ans1)

# Part 2
ans2 = 1
for line in real_data[:3]:
    m_costs = parse_blueprint(line)
    m_max = get_max_robs(m_costs)
    ans2 *= dfs(m_max, m_costs, {}, [0, 0, 0, 0], [1, 0, 0, 0], 32)
print(ans2)
