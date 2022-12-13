from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class State:
    x: int
    pos: int


@dataclass
class Instr:
    cycles: int
    code: str

    @classmethod
    def from_str(cls, x: str) -> "Instr":
        if x.startswith("addx"):
            return Instr(2, x)
        return Instr(1, x)

    def execute(self, state: State) -> None:
        if self.code == "noop":
            return
        if self.code.startswith("addx"):
            operands = self.code.split(" ")
            state.x += int(operands[1])


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


_state = State(x=1, pos=0)
test_data = Path("inputs/10.txt").read_text()


def interesting_cycle(x: int) -> bool:
    return x > 0 and ((x - 20) % 40 == 0)


def is_lit(state: State) -> bool:
    return (
        state.pos == state.x
        or state.pos == state.x - 1
        or state.pos == state.x + 1
    )


cpu = CPU(instructions=deque(map(Instr.from_str, test_data.splitlines())))
sig_sum = 0

for cycle in range(1, 243):
    if interesting_cycle(cycle):
        sig_sum += cycle * _state.x

    if _state.pos == 0:
        print()

    print("#" if is_lit(_state) else " ", end="")

    cpu.tick(_state)

print("\nsignal sum:", sig_sum)
