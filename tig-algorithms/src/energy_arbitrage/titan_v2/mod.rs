use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

#[allow(
    dead_code,
    unused_variables,
    unused_assignments,
    unused_mut,
    non_snake_case
)]
mod sub_t49;
#[allow(
    dead_code,
    unused_variables,
    unused_assignments,
    unused_mut,
    non_snake_case
)]
mod sub_t50;
#[allow(
    dead_code,
    unused_variables,
    unused_assignments,
    unused_mut,
    non_snake_case
)]
mod sub_t51;
#[allow(
    dead_code,
    unused_variables,
    unused_assignments,
    unused_mut,
    non_snake_case
)]
mod sub_t52;
#[allow(
    dead_code,
    unused_variables,
    unused_assignments,
    unused_mut,
    non_snake_case
)]
mod sub_t53;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match challenge.num_batteries {
        n if n <= 15 => sub_t49::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 30 => sub_t50::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 50 => sub_t51::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 80 => sub_t52::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 150 => sub_t53::solve_challenge(challenge, save_solution, hyperparameters),
        n => Err(anyhow!("titan_v2: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("titan_v2");
}
