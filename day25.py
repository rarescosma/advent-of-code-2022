from pathlib import Path
from textwrap import dedent

test_data = dedent(
    """1=-0-2
12111
2=0=
21
2=01
111
20012
112
1=-1=
1-12
12
1=
122
""".strip()
).splitlines()

real_data = Path("inputs/25.txt").read_text().splitlines()
real: bool = True

the_data = real_data if real else test_data

snafu = {"0": 0, "1": 1, "2": 2, "-": -1, "=": -2}
ufans = {v: k for k, v in snafu.items()}

total: int = 0
for number in the_data:
    power: int = 1
    for digit in number[::-1]:
        total += snafu[digit] * power
        power *= 5

output: str = ""
while total:
    total, rem = divmod(total, 5)
    output = ufans[((rem + 2) % 5) - 2] + output
    if rem > 2:
        total += 1

print(output)
