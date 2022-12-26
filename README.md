# Advent of Code 2022 :christmas_tree:
Solutions to all 25 Advent of Code 2022 puzzles mostly written in Python (in an attempt to get back some Python foo) save for days [12](2022/src/bin/day12.rs) and [24](2022/src/bin/day24.rs) which involve [Dijsktra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) so had to be done in Rust to re-use the [generic implementation](https://github.com/rarescosma/advent-of-code-2021-rs/blob/main/crates/aoc_dijsktra/src/lib.rs) from last year. :snake: :crab:

Thanks [@ericwastl](https://twitter.com/ericwastl) for providing the challenge.

## Usage
```sh
./run.sh   # sequential run of all days
./par.sh   # parallel run of all days
./run.sh [INTERPETER] # sequential run of all days using specified INTERPRETER (eg. pypy)
./par.sh [INTERPETER] # parallel run of all days using specified INTERPRETER (eg. pypy)
```
