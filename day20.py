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


def mix_it(the_data: list[str], coef: int = 1, times: int = 1) -> El:
    dll = [El(int(_) * coef) for _ in the_data]
    zero = dll[the_data.index("0")]

    # link the doubly linked list
    for i, el in enumerate(dll):
        el.left = dll[(i - 1) % len(dll)]
        el.right = dll[(i + 1) % len(dll)]

    modulus = len(dll) - 1
    for _ in range(times):
        for el in dll:
            if el.k > 0:
                dest = el
                for _ in range(el.k % modulus):
                    dest = cast(El, dest.right)
                n_dest = dest.right
                intercede(el, dest, n_dest)
            elif el.k < 0:
                dest = el
                for _ in range(-el.k % modulus):
                    dest = cast(El, dest.left)
                n_dest = dest.left
                intercede(el, n_dest, dest)
    return cast(El, zero)


def solve(offset: int, times: int) -> int:
    _ans = 0
    _zero = mix_it(real_data, offset, times)
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
