use aoc_2dmap::prelude::*;
use aoc_dijsktra::{Dijsktra, GameState, Transform};
use aoc_prelude::*;

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash, Clone)]
struct State {
    pos: Pos,
}

struct Context {
    map: Map<u8>,
    start_pos: Pos,
    min_cost: usize,
}

impl GameState<Context> for State {
    type Steps = ArrayVec<Pos, 4>;

    fn accept(&self, cost: usize, context: &mut Context) -> bool {
        if context.map.get_unchecked(self.pos) == b'a' && cost < context.min_cost {
            context.min_cost = cost;
        }
        if self.pos == context.start_pos {
            return true;
        }
        false
    }

    fn steps(&self, context: &mut Context) -> Self::Steps {
        let mut steps = ArrayVec::new();

        let current_val = context.map.get_unchecked(self.pos);

        for n_pos in self.pos.neighbors_simple() {
            if let Some(dest_val) = context.map.get(n_pos) {
                // going backwards from End we can climb as much as we want
                // but only descend as most 1 height
                if dest_val >= current_val || current_val - dest_val == 1 {
                    steps.push(n_pos)
                }
            }
        }
        steps
    }
}

impl Transform<State> for Pos {
    fn cost(&self) -> usize {
        1
    }

    fn transform(&self, _state: &State) -> State {
        State { pos: *self }
    }
}

fn read_input() -> Vec<&'static str> {
    include_str!("../../../inputs/12.txt").lines().collect()
}

fn main() {
    let now = std::time::Instant::now();

    let input = read_input();

    let map_size = (input[0].len(), input.len());

    let mut map = Map::<u8>::new(map_size, input.into_iter().flat_map(|l| l.bytes()));

    let mut end_pos = Pos::default();
    let mut start_pos = Pos::default();

    for pos in map.iter().collect::<Vec<_>>().into_iter() {
        let tile = map.get_unchecked(pos);
        if tile == b'S' {
            start_pos = pos;
            map.set(pos, b'a');
        }
        if tile == b'E' {
            end_pos = pos;
            map.set(pos, b'z');
        }
    }

    let mut ctx = Context {
        map,
        start_pos,
        min_cost: usize::MAX,
    };
    let start_cost = State { pos: end_pos }.dijsktra(&mut ctx);

    // Part 1
    println!("{}", start_cost.unwrap());

    // Part 2
    println!("{}", ctx.min_cost);

    let time = now.elapsed().as_micros();
    eprintln!("Time: {}us", time);
}
