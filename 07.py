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

    def up(self) -> "State":
        return replace(self, cur_dir=self.cur_dir.parent)

    def chdir(self, child: str) -> "State":
        if child == "/":
            return replace(self, cur_dir=PurePath("/"))

        new_dir = self.cur_dir / child
        dir_name = new_dir.as_posix()

        if dir_name not in self.dirs:
            new_dirs = {**self.dirs, dir_name: 0}
            return replace(self, cur_dir=new_dir, dirs=new_dirs)
        return self

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
        cur_path = path
        while cur_path.parent != cur_path:
            yield cur_path.as_posix()
            cur_path = cur_path.parent
        yield "/"


def next_command(lines: list[str]) -> Generator[Command, None, None]:
    last_cmd = Command("", [], [])

    for line in lines:
        if line.startswith("$"):
            if last_cmd.cmd:
                yield last_cmd

            parts = line.split(" ")
            parts.pop(0)  # throw the $ away

            _cmd, *args = parts
            last_cmd = Command(_cmd, args, [])
        else:
            last_cmd = replace(last_cmd, output=[*last_cmd.output, line])

    yield last_cmd


def update_state(_state: State, _cmd: Command) -> State:
    if _cmd.cmd == "cd":
        if _cmd.args[0] == "..":
            return _state.up()
        else:
            return _state.chdir(_cmd.args[0])
    elif _cmd.cmd == "ls":
        return _state.update_sizes(_cmd.output)

    return _state


state = State(PurePath("/"), {"/": 0})
for cmd in next_command(Path("inputs/07.txt").read_text().splitlines()):
    state = update_state(state, cmd)

answer = 0
for _, v in state.dirs.items():
    if v <= 100000:
        answer += v

print("Part 1 answer:", answer)

tot_size = 70000000 - state.dirs["/"]
needed = 30000000
least_dir = 100000000
for _, v in state.dirs.items():
    if v < least_dir and (tot_size + v) >= needed:
        least_dir = v

print("\nPart 2 answer:", least_dir)
