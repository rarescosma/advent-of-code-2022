use std::{
    cmp::{max, min},
    collections::VecDeque,
    fs,
};

use ahash::HashSet;
use aoc_prelude::lazy_static;
use rayon::prelude::*;
use regex::Regex;

#[derive(Debug)]
struct CostItem {
    amount: usize,
    resource_type: usize,
}

type Bots = [usize; 4];
type Res = [usize; 4];

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Clone, Debug)]
struct Wunit {
    res: Res,
    bots: Bots,
    time: usize,
}

const GEODE: usize = 3;

fn parse_blueprint(blueprint: &str) -> Vec<Vec<CostItem>> {
    lazy_static! {
        static ref COST_RE: Regex = Regex::new(r"\d+\s[a-z]+").unwrap();
        static ref RES: Vec<&'static str> = vec!["ore", "clay", "obsidian"];
    }
    let mut res = Vec::new();
    for recipe in blueprint.split(".") {
        let mut inner = Vec::new();
        for m in COST_RE.find_iter(recipe) {
            let mut parts = m.as_str().split(" ");
            let amount = parts
                .next()
                .map(|x| x.parse::<usize>().expect("invalid amount"))
                .unwrap();
            let resource_type = parts.next().unwrap();
            let resource_type = RES
                .iter()
                .position(|x| *x == resource_type)
                .expect("invalid resource");
            inner.push(CostItem {
                amount,
                resource_type,
            })
        }
        if !inner.is_empty() {
            res.push(inner);
        }
    }
    res
}

fn wait_time(costs: &Vec<CostItem>, res: Res, bots: Bots) -> usize {
    #[inline(always)]
    fn div_up(a: usize, b: usize) -> usize {
        (a + (b - 1)) / b
    }

    let mut max_wait = 0;
    for cost in costs {
        if bots[cost.resource_type] == 0 {
            return 2 << 8;
        }
        if res[cost.resource_type] <= cost.amount {
            max_wait = max(
                max_wait,
                div_up(
                    cost.amount - res[cost.resource_type],
                    bots[cost.resource_type],
                ),
            )
        }
    }
    max_wait
}

fn get_max_bots(costs: &[Vec<CostItem>]) -> [usize; 3] {
    let mut max_bots = [0, 0, 0];
    for blueprint in costs.iter() {
        for cost_item in blueprint.iter() {
            max_bots[cost_item.resource_type] =
                max(max_bots[cost_item.resource_type], cost_item.amount);
        }
    }
    max_bots
}

fn dfs(max_bots: [usize; 3], costs: &[Vec<CostItem>], time: usize) -> usize {
    let mut seen: HashSet<Wunit> = HashSet::default();
    let mut queue: VecDeque<Wunit> = VecDeque::default();
    queue.push_back(Wunit {
        res: [0, 0, 0, 0],
        bots: [1, 0, 0, 0],
        time,
    });

    let mut best = 0;

    while !queue.is_empty() {
        let wunit = queue.pop_front().unwrap();
        best = max(best, wunit.res[GEODE] + wunit.bots[GEODE] * wunit.time);

        for (bot_type, blueprint) in costs.iter().enumerate() {
            // Obs 1: do not build more than the max_bots number of bots
            // for each type (except geode)
            if bot_type != GEODE && wunit.bots[bot_type] >= max_bots[bot_type] {
                continue;
            }

            let wait = wait_time(blueprint, wunit.res, wunit.bots);

            if wait + 1 >= wunit.time {
                continue;
            }
            let _t = wunit.time - wait - 1;
            let mut _bots = wunit.bots;
            let mut _res_slice: [usize; 4] = wunit
                .res
                .iter()
                .zip(wunit.bots.iter())
                .map(|(old, rate)| old + rate * (wait + 1))
                .collect::<Vec<_>>()
                .try_into()
                .expect("nope");
            _bots[bot_type] += 1;

            for b in blueprint.iter() {
                _res_slice[b.resource_type] -= b.amount;
            }

            // Obs 2: throw away extra resources
            for res_type in 0..3 {
                _res_slice[res_type] = min(_res_slice[res_type], max_bots[res_type] * _t)
            }
            let _k = Wunit {
                res: _res_slice,
                bots: _bots,
                time: _t,
            };
            if !seen.contains(&_k) {
                seen.insert(_k.clone());
                queue.push_back(_k);
            }
        }
    }

    best
}

fn solve(line: &str, start_time: usize) -> usize {
    let costs = parse_blueprint(line);
    dfs(get_max_bots(&costs), &costs, start_time)
}

fn main() {
    let now = std::time::Instant::now();

    let file_path = "inputs/19.txt";
    let input: Vec<String> = fs::read_to_string(file_path)
        .expect("valid input file")
        .lines()
        .map(String::from)
        .collect();

    let p1: usize = input
        .clone()
        .into_par_iter()
        .map(|x| solve(&x, 24))
        .enumerate()
        .map(|(i, v)| (i + 1) * v)
        .sum();
    println!("{p1}");

    let input_2 = input.iter().take(3).collect::<Vec<_>>();
    let p2: usize = input_2.into_par_iter().map(|x| solve(x, 32)).product();
    println!("{p2}");

    let time = now.elapsed().as_micros();
    eprintln!("Time: {}us", time);
}
