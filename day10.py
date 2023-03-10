from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class State:
    x_pos: int
    pos: int


@dataclass
class Instr:
    cycles: int
    code: str

    @classmethod
    def from_str(cls, _x: str) -> "Instr":
        if _x.startswith("addx"):
            return cls(2, _x)
        return cls(1, _x)

    def execute(self, state: State) -> None:
        if self.code == "noop":
            return
        if self.code.startswith("addx"):
            operands = self.code.split(" ")
            state.x_pos += int(operands[1])


@dataclass
class CPU:
    instructions: deque[Instr]
    cur_instr: Optional[Instr] = None

    def tick(self, state: State) -> None:
        if self.cur_instr is None:
            if not self.instructions:
                return
            self.cur_instr = self.instructions.popleft()

        pos = state.pos + 1
        state.pos = pos % 40

        self.cur_instr.cycles -= 1
        if self.cur_instr.cycles == 0:
            self.cur_instr.execute(state)
            self.cur_instr = None


_state = State(x_pos=1, pos=0)
test_data = Path("inputs/10.txt").read_text()


def interesting_cycle(_x: int) -> bool:
    return _x > 0 and ((_x - 20) % 40 == 0)


def is_lit(state: State) -> bool:
    return state.pos in (state.x_pos, state.x_pos - 1, state.x_pos + 1)


cpu = CPU(instructions=deque(map(Instr.from_str, test_data.splitlines())))
sig_sum: int = 0
display: str = ""

for cycle in range(1, 241):
    if interesting_cycle(cycle):
        sig_sum += cycle * _state.x_pos

    if _state.pos == 0:
        display += "\n"

    display += "#" if is_lit(_state) else " "

    cpu.tick(_state)

# Part 1
print(sig_sum)

# Part 2
print(display)
