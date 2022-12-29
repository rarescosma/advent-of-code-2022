from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Generator


@dataclass(frozen=True)
class Command:
    cmd: str
    args: list[str]
    output: list[str]


@dataclass(frozen=True)
class State:
    cur_dir: PurePath
    dirs: dict[str, int]

    def go_up(self) -> "State":
        return replace(self, cur_dir=self.cur_dir.parent)

    def chdir(self, child: str) -> "State":
        if child == "/":
            return replace(self, cur_dir=PurePath("/"))

        return replace(self, cur_dir=(self.cur_dir / child))

    def update_sizes(self, ls_output: list[str]) -> "State":
        new_dirs = self.dirs
        for out in ls_output:
            out_parts = out.split(" ")
            if out_parts and out_parts[0].isdigit():
                for asc in self.get_ascendants(self.cur_dir):
                    new_dirs[asc] += int(out_parts[0])
        return replace(self, dirs=new_dirs)

    @staticmethod
    def get_ascendants(path: PurePath) -> Generator[str, None, None]:
        yield path.as_posix()
        for par in path.parents:
            yield par.as_posix()


def next_command(lines: list[str]) -> Generator[Command, None, None]:
    last_cmd = Command("", [], [])

    for line in lines:
        if line.startswith("$"):
            if last_cmd.cmd:
                yield last_cmd

            _cmd, *args = line.split(" ")[1:]  # throw the $ away
            last_cmd = Command(_cmd, args, [])
        else:
            last_cmd = replace(last_cmd, output=[*last_cmd.output, line])

    yield last_cmd


def update_state(_state: State, _cmd: Command) -> State:
    if _cmd.cmd == "cd":
        if _cmd.args[0] == "..":
            return _state.go_up()
        return _state.chdir(_cmd.args[0])
    if _cmd.cmd == "ls":
        return _state.update_sizes(_cmd.output)

    return _state


state = State(PurePath("/"), defaultdict(int))
for cmd in next_command(Path("inputs/07.txt").read_text().splitlines()):
    state = update_state(state, cmd)

# Part 1
a1: int = 0
for _, v in state.dirs.items():
    if v <= 100000:
        a1 += v
print(a1)

# Part 2
tot_size = 70000000 - state.dirs["/"]
needed: int = 30000000
a2: int = int(1e12)
for _, v in state.dirs.items():
    if v < a2 and (tot_size + v) >= needed:
        a2 = v
print(a2)
