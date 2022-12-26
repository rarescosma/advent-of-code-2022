from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Optional, cast

real_data = Path("inputs/20.txt").read_text().splitlines()
test_data = dedent(
    """
1
2
-3
3
-2
0
4
""".strip()
).splitlines()
real: bool = True

the_data = real_data if real else test_data


@dataclass
class El:
    k: int
    right: "Ptr" = None
    left: "Ptr" = None


Ptr = Optional[El]


def intercede(_el: El, bef: Ptr, aft: Ptr) -> None:
    if bef is None or aft is None:
        return
    # stich: left <-> el <-> right ---> left <-> right
    cast(El, _el.left).right = _el.right
    cast(El, _el.right).left = _el.left

    # inter: dest <-> n_dest ---> dest <-> el <-> n_dest
    bef.right = _el
    _el.right = aft
    aft.left = _el
    _el.left = bef


def mix_it(_the_data: list[str], coef: int = 1, times: int = 1) -> El:
    dll = [El(int(_) * coef) for _ in _the_data]
    zero = dll[_the_data.index("0")]

    # link the doubly linked list
    for i, node in enumerate(dll):
        node.left = dll[(i - 1) % len(dll)]
        node.right = dll[(i + 1) % len(dll)]

    modulus = len(dll) - 1
    for _ in range(times):
        for node in dll:
            if node.k > 0:
                dest = node
                for _ in range(node.k % modulus):
                    dest = cast(El, dest.right)
                n_dest = dest.right
                intercede(node, dest, n_dest)
            elif node.k < 0:
                dest = node
                for _ in range(-node.k % modulus):
                    dest = cast(El, dest.left)
                n_dest = dest.left
                intercede(node, n_dest, dest)
    return cast(El, zero)


def solve(offset: int, times: int) -> int:
    _ans = 0
    _zero = mix_it(the_data, offset, times)
    _el = _zero
    for _ in range(3):
        for __ in range(1000):
            _el = cast(El, _el.right)
        _ans += _el.k
    return _ans


# Part 1
print(solve(1, 1))

# Part 2
print(solve(811589153, 10))
