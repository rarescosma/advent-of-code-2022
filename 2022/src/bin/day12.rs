use std::fmt::Debug;
use std::hash::Hash;

use aoc_2dmap::prelude::*;
use aoc_dijsktra::{Dijsktra, GameState, Transform};
use aoc_prelude::*;

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash, Clone)]
struct State {
    pos: Pos,
}

#[derive(Debug)]
struct Move {
    to: Pos,
}

#[derive(Copy, Clone)]
enum Tile {
    Start,
    End,
    Node(u8),
}

impl Tile {
    pub fn val(&self) -> u8 {
        match self {
            Tile::Node(x) => *x,
            Tile::Start => b'a',
            Tile::End => b'z',
        }
    }
}

struct Context {
    map: Map<Tile>,
    goals: HashSet<Pos>,
}

impl GameState<Context> for State {
    type Steps = ArrayVec<Move, 4>;

    fn accept(&self, context: &mut Context) -> bool {
        context.goals.contains(&self.pos)
    }

    fn steps(&self, context: &mut Context) -> Self::Steps {
        let mut steps = ArrayVec::new();

        let current_val = context.map.get_unchecked(self.pos).val();

        for n_pos in self.pos.neighbors_simple() {
            if let Some(tile) = context.map.get(n_pos) {
                let dest_val = tile.val();
                // going backwards from End we can climb as much as we want
                // but only descend as most 1 height
                if dest_val >= current_val || current_val - dest_val == 1 {
                    steps.push(Move { to: n_pos })
                }
            }
        }
        steps
    }
}

impl Transform<State> for Move {
    fn cost(&self) -> usize {
        1
    }

    fn transform(&self, _state: &State) -> State {
        State { pos: self.to }
    }
}

fn read_input() -> Vec<&'static str> {
    include_str!("../../../inputs/12.txt").lines().collect()
}

impl From<u8> for Tile {
    fn from(c: u8) -> Self {
        match c {
            b'S' => Tile::Start,
            b'E' => Tile::End,
            x => Tile::Node(x),
        }
    }
}

fn main() {
    let now = std::time::Instant::now();

    let input = read_input();

    let map_size = (input[0].len(), input.len());

    let map = Map::<Tile>::new(
        map_size,
        input.into_iter().flat_map(|l| l.bytes().map(Tile::from)),
    );

    let mut end_pos = Pos::default();
    let mut start_pos = Pos::default();
    let a_val = b'a';
    let mut a_positions = HashSet::default();

    for pos in map.iter() {
        let tile = map.get_unchecked(pos);
        if tile.val() == a_val {
            a_positions.insert(pos);
        }
        if matches!(tile, Tile::Start) {
            start_pos = pos;
        }
        if matches!(tile, Tile::End) {
            end_pos = pos;
        }
    }

    // Part 1
    let mut c1 = Context {
        map: map.clone(),
        goals: HashSet::from_iter([start_pos]),
    };
    let ans1 = State { pos: end_pos }.dijsktra(&mut c1);
    println!("part 1 steps: {:?}", ans1);

    // Part 2
    let mut c2 = Context {
        map,
        goals: a_positions,
    };
    let ans2 = State { pos: end_pos }.dijsktra(&mut c2);
    println!("part 2 steps: {:?}", ans2);

    let time = now.elapsed().as_micros();
    println!("Time: {}us", time);
}
