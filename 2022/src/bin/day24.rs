use aoc_2dmap::prelude::*;
use aoc_dijsktra::{Dijsktra, GameState, Transform};
use aoc_prelude::*;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialOrd, Ord, Eq, PartialEq, Hash)]
enum Tile {
    Wall,
    Empty,
    Blizz(ArrayVec<u8, 4>),
}

impl Display for Tile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Tile::Wall => write!(f, "#"),
            Tile::Empty => write!(f, "."),
            Tile::Blizz(bs) => {
                if bs.len() > 1 {
                    write!(f, "{}", bs.len())
                } else {
                    write!(f, "{}", std::str::from_utf8(&[bs[0]]).unwrap())
                }
            }
        }
    }
}

impl From<u8> for Tile {
    fn from(c: u8) -> Self {
        match c {
            b'#' => Self::Wall,
            b'.' => Self::Empty,
            x => Self::Blizz(ArrayVec::from_iter(std::iter::once(x))),
        }
    }
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash, Clone)]
struct State {
    pos: Pos,
    time: usize,
}

struct Ctx {
    start_pos: Pos,
    end_pos: Pos,
    end_time: usize,
    map_stages: ArrayVec<Map<Tile>, 300>,
}

struct Move(Pos);

impl Transform<State> for Move {
    fn cost(&self) -> usize {
        1
    }

    fn transform(&self, game_state: &State) -> State {
        State {
            pos: self.0,
            time: game_state.time + 1,
        }
    }
}

impl GameState<Ctx> for State {
    type Steps = ArrayVec<Move, 5>;

    fn accept(&self, _cost: usize, ctx: &mut Ctx) -> bool {
        if self.pos == ctx.end_pos {
            ctx.end_time = self.time;
            return true;
        }
        false
    }

    fn steps(&self, ctx: &mut Ctx) -> Self::Steps {
        let mut steps = ArrayVec::new();

        let current_pos = self.pos;
        let next_map = &ctx.map_stages[(self.time + 1) % 300];

        if current_pos == ctx.start_pos {
            steps.push(Move(current_pos));
        }

        for n_pos in self
            .pos
            .neighbors_simple()
            .chain(std::iter::once(current_pos))
        {
            match next_map.get(n_pos) {
                None => {
                    if n_pos == ctx.end_pos {
                        steps.push(Move(n_pos))
                    }
                }
                Some(Tile::Empty) => steps.push(Move(n_pos)),
                _ => {}
            }
        }

        steps
    }
}

fn read_input() -> Vec<&'static str> {
    include_str!("../../../inputs/24.txt").lines().collect()
}

fn step(map: &Map<Tile>) -> Map<Tile> {
    let mut new_map = Map::fill(map.size, Tile::Empty);
    for pos in map.iter() {
        match map.get_unchecked(pos) {
            Tile::Blizz(bs) => {
                for b in bs {
                    let mut np = match b {
                        b'^' => pos + (0, -1).into(),
                        b'<' => pos + (-1, 0).into(),
                        b'v' => pos + (0, 1).into(),
                        b'>' => pos + (1, 0).into(),
                        _ => continue,
                    };
                    if np.x < 0 {
                        np.x = map.size.x - 1
                    }
                    if np.x == map.size.x {
                        np.x = 0
                    }
                    if np.y < 0 {
                        np.y = map.size.y - 1
                    }
                    if np.y == map.size.y {
                        np.y = 0
                    }
                    match new_map.get_unchecked_ref(np) {
                        Tile::Blizz(nbs) => {
                            nbs.push(b);
                        }
                        _ => new_map.set(np, Tile::Blizz(ArrayVec::from_iter(std::iter::once(b)))),
                    }
                }
            }
            Tile::Wall | Tile::Empty => {}
        }
    }
    new_map
}

fn main() {
    let input = read_input();

    let i_len = input.len();
    let map_size = (input[0].len() - 2, i_len - 2);

    let mut map = Map::<Tile>::new(
        map_size,
        input[1..i_len - 1]
            .iter()
            .flat_map(|t| t.bytes().dropping(1).dropping_back(1).map(Tile::from)),
    );

    let mut seen: HashSet<Map<Tile>> = HashSet::new();
    let mut map_stages = ArrayVec::<Map<Tile>, 300>::new();
    map_stages.push(map.clone());
    seen.insert(map.clone());
    loop {
        let new_map = step(&map);
        if seen.contains(&new_map) {
            break;
        }
        map = new_map.clone();
        seen.insert(new_map.clone());
        map_stages.push(new_map);
    }

    let start_pos: Pos = (0, -1).into();
    let end_pos: Pos = Pos::from((map_size.0 - 1, map_size.1));

    // Part 1 - there
    let mut ctx1 = Ctx {
        start_pos,
        end_pos,
        end_time: 0,
        map_stages: map_stages.clone(),
    };
    let st1 = State {
        pos: start_pos,
        time: 0,
    };
    let mut ans = st1.dijsktra(&mut ctx1).unwrap();
    println!("{}", ans);

    // Part 2 - and back again
    let mut ctx2 = Ctx {
        start_pos: end_pos,
        end_pos: start_pos,
        end_time: 0,
        map_stages: map_stages.clone(),
    };

    let st2 = State {
        pos: end_pos,
        time: ctx1.end_time,
    };
    ans += st2.dijsktra(&mut ctx2).unwrap();

    let mut ctx3 = Ctx {
        start_pos,
        end_pos,
        end_time: 0,
        map_stages,
    };

    let st3 = State {
        pos: start_pos,
        time: ctx2.end_time,
    };
    ans += st3.dijsktra(&mut ctx3).unwrap();
    println!("{}", ans);
}