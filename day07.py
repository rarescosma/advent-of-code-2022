from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Generator


@dataclass(frozen=True)
class Command:
    cmd: str
    args: list[str]
    output: list[str]

    def add_output_line(self, line: str) -> "Command":
        return replace(self, output=[*self.output, line])


@dataclass(frozen=True)
class State:
    cur_dir: PurePath
    dirs: dict[str, int]

    def update_state(self, from_cmd: Command) -> "State":
        if from_cmd.cmd == "cd":
            return self.chdir(from_cmd.args[0])
        if from_cmd.cmd == "ls":
            return self.update_sizes(from_cmd.output)
        return self

    def chdir(self, child: str) -> "State":
        if child == "/":
            return replace(self, cur_dir=PurePath("/"))
        if child == "..":
            return replace(self, cur_dir=self.cur_dir.parent)
        return replace(self, cur_dir=(self.cur_dir / child))

    def update_sizes(self, ls_output: list[str]) -> "State":
        new_dirs = self.dirs
        f_sizes = sum(
            int(first)
            for line in ls_output
            if (first := line.split(" ")[0]).isnumeric()
        )
        for asc in self.get_ascendants(self.cur_dir):
            new_dirs[asc] += int(f_sizes)
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
            last_cmd = last_cmd.add_output_line(line)

    yield last_cmd


the_data = Path("inputs/07.txt").read_text().splitlines()

state = State(PurePath("/"), defaultdict(int))
for cmd in next_command(the_data):
    state = state.update_state(cmd)

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
