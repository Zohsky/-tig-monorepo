use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::energy_arbitrage::*;

pub mod track_t49;
pub mod track_t50;
pub mod t51_engine;
pub mod t52_engine;
pub mod t53_engine;

/// Transaction-cost wedges for a complete charge/discharge cycle.
///
/// A MWh bought for charging yields `eta_rt` MWh at discharge, so the buy-side
/// threshold carries `kappa * (1 + eta_rt)`. Expressed per discharged MWh, the
/// sell-side threshold carries `kappa * (1 + 1 / eta_rt)`.
#[inline(always)]
fn round_trip_transaction_friction(eta_rt: f64, kappa: f64) -> (f64, f64) {
    debug_assert!(eta_rt > 0.0 && eta_rt <= 1.0);
    #[cfg(test)]
    if std::env::var_os("TIG_BENCH_LEGACY_FRICTION").is_some() {
        return (2.0 * kappa, 2.0 * kappa);
    }
    (kappa * (1.0 + eta_rt), kappa * (1.0 + 1.0 / eta_rt))
}

#[cfg(test)]
mod tests {
    use super::round_trip_transaction_friction;
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn round_trip_friction_is_symmetric_only_at_unit_efficiency() {
        let (charge, discharge) = round_trip_transaction_friction(1.0, 0.25);
        assert_eq!(charge, 0.5);
        assert_eq!(discharge, 0.5);

        let (charge, discharge) = round_trip_transaction_friction(0.81, 0.25);
        assert!((charge - 0.4525).abs() < 1e-12);
        assert!((discharge - 0.558641975308642).abs() < 1e-12);
        assert!(charge < 0.5);
        assert!(discharge > 0.5);
    }

    #[test]
    #[ignore = "paired benchmark harness; run explicitly with TIG_BENCH_* variables"]
    fn paired_benchmark_case() {
        let scenario = match std::env::var("TIG_BENCH_TRACK").unwrap().as_str() {
            "T49" => Scenario::BASELINE,
            "T50" => Scenario::CONGESTED,
            "T51" => Scenario::MULTIDAY,
            "T52" => Scenario::DENSE,
            "T53" => Scenario::CAPSTONE,
            track => panic!("unsupported TIG_BENCH_TRACK={track}"),
        };
        let nonce: u64 = std::env::var("TIG_BENCH_NONCE")
            .unwrap()
            .parse()
            .unwrap();
        let mut seed = [0_u8; 32];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = (nonce as u8).wrapping_add((i as u8).wrapping_mul(37));
        }
        let challenge = Challenge::generate_instance(&seed, &Track { s: scenario }).unwrap();
        let saved = Mutex::new(None::<Solution>);
        let started = std::time::Instant::now();
        solve_challenge(
            &challenge,
            &|solution| {
                *saved.lock().unwrap() = Some(solution.clone());
                Ok(())
            },
            &None,
        )
        .unwrap();
        let elapsed = started.elapsed();
        let solution = saved.into_inner().unwrap().expect("solver saved no solution");
        let output = std::env::var("TIG_BENCH_OUTPUT").unwrap();
        std::fs::write(&output, serde_json::to_vec(&solution).unwrap()).unwrap();
        println!(
            "paired_case track={} nonce={} elapsed_ms={} steps={} output={}",
            std::env::var("TIG_BENCH_TRACK").unwrap(),
            nonce,
            elapsed.as_millis(),
            solution.schedule.len(),
            output,
        );
    }
}

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

fn n(v: u64) -> Value { Value::Number(Number::from(v)) }
fn f(v: f64) -> Value { Value::Number(Number::from_f64(v).unwrap()) }
fn b(v: bool) -> Value { Value::Bool(v) }

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // titan_v6: per-track solvers assembled from the best valid iter of each track
    // (T49=t49/i48, T50=t50/i86, T51=t51/i90, T52=t52/i76, T53=t53/i87).
    // Baked HP = the iter's own baked defaults merged with the winning bench override,
    // so hp_json={} reproduces the winning per-track Q. User HP always win (merge_hp).
    match challenge.num_batteries {
        n if n <= 15 => {
            let hp = merge_hp(hyperparameters, vec![
                ("soc_levels", self::n(101)),
                ("action_grid", self::n(40)),
                ("asca_iters", self::n(25)),
                ("ternary_iters", self::n(25)),
                ("convergence_tol", f(1e-4)),
                ("k_clusters", self::n(80)),
                ("deflator_iters", self::n(15)),
                ("lp_refine_sweeps", self::n(3)),
                ("cg_iters", self::n(20)),
                ("use_lp", b(true)),
                ("use_sdp", b(true)),
                ("use_cg", b(true)),
                ("network_derating", f(1.00)),
                ("use_analytical_pricing", b(true)),
                ("use_pce_affine_recourse", b(true)),
            ]);
            track_t49::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 30 => {
            let hp = merge_hp(hyperparameters, vec![
                ("soc_levels", self::n(201)),
                ("action_grid", self::n(40)),
                ("asca_iters", self::n(4)),
                ("ternary_iters", self::n(20)),
                ("deflator_iters", self::n(50)),
                ("network_derating", f(0.35)),
                ("max_admm_iters", self::n(10)),
                ("lp_total_pivots", self::n(15000)),
                ("dw_total_pivot_budget", self::n(8000)),
                ("lns_lp_pivots_total", self::n(6000)),
                ("use_lp", b(true)),
                ("use_dw", b(true)),
                ("use_lns", b(true)),
                ("use_kkt", b(true)),
                ("ct_step_eta", f(0.5)),
                ("dp_rho_jump", f(0.015)),
                ("ct_gdd_alpha", f(1.0)),
                ("lmp_threshold", f(0.65)),
                ("anticipate_lmp", b(true)),
                ("lmp_premium_scale", f(1.2)),
                ("use_primal_refine", b(true)),
                ("dw_mu_damping_alpha", f(0.6)),
                ("use_lmp_premiums_kkt", b(true)),
                ("use_ct_adaptive_per_line", b(true)),
                ("use_ptdf_constraint_tracking", b(true)),
            ]);
            track_t50::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 50 => {
            let hp = merge_hp(hyperparameters, vec![
                ("dp_soc_levels", self::n(97)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("grad_outer_iters", self::n(80)),
                ("grad_ls_iters", self::n(6)),
                ("bisect_iters", self::n(30)),
                ("coord_polish_passes", self::n(2)),
                ("lookahead_horizon", self::n(24)),
                ("rh_stride", self::n(3)),
                ("pga_beta_end", f(0.6)),
                ("use_momentum", b(true)),
                ("use_bb_clamps", b(true)),
                ("soc_ref_lambda", f(0.05)),
                ("use_admm_solver", b(false)),
                ("use_cosine_beta", b(true)),
                ("soc_ref_dyn_stride", self::n(6)),
                ("joint_triplet_top_k", self::n(15)),
                ("use_rolling_horizon", b(true)),
                ("joint_triplet_budget", self::n(300)),
                ("use_joint_pair_polish", b(true)),
                ("use_joint_triplet_polish", b(true)),
            ]);
            t51_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 80 => {
            let hp = merge_hp(hyperparameters, vec![
                ("dp_soc_levels", self::n(65)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("grad_outer_iters", self::n(75)),
                ("grad_ls_iters", self::n(12)),
                ("bisect_iters", self::n(30)),
                ("coord_polish_passes", self::n(0)),
                ("lookahead_horizon", self::n(24)),
                ("cwv_lambda", f(0.25)),
                ("ct_step_eta", f(1.0)),
                ("use_dp_seed", b(false)),
                ("use_ptdf_ct", b(true)),
                ("ct_ref_kappa", f(0.0)),
                ("cwv_clusters", self::n(4)),
                ("use_momentum", b(true)),
                ("lmp_threshold", f(0.5)),
                ("lr_growth_cap", f(1.025)),
                ("use_bb_clamps", b(true)),
                ("use_zero_seed", b(false)),
                ("anticipate_lmp", b(true)),
                ("cwv_agg_levels", self::n(65)),
                ("use_cosine_beta", b(true)),
                ("use_composite_wv", b(true)),
                ("use_pwl_value_dp", b(false)),
                ("joint_pair_budget", self::n(1024)),
                ("lmp_premium_scale", f(2.0)),
                ("pwl_max_breakpoints", self::n(64)),
                ("congestion_grid_alpha", f(0.0)),
                ("use_joint_pair_polish", b(true)),
            ]);
            t52_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 150 => {
            let hp = merge_hp(hyperparameters, vec![
                ("dp_soc_levels", self::n(65)),
                ("dp_action_levels", self::n(9)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(50)),
                ("grad_outer_iters", self::n(28)),
                ("grad_ls_iters", self::n(6)),
                ("bisect_iters", self::n(30)),
                ("coord_polish_passes", self::n(1)),
                ("lookahead_horizon", self::n(24)),
                ("num_seeds", self::n(1)),
                ("cwv_lambda", f(0.3)),
                ("ct_step_eta", f(0.5)),
                ("use_ptdf_ct", b(true)),
                ("ct_gdd_alpha", f(0.0)),
                ("ct_ref_kappa", f(0.0)),
                ("use_momentum", b(true)),
                ("anticipate_lmp", b(true)),
                ("use_cosine_beta", b(false)),
                ("use_composite_wv", b(true)),
                ("joint_pair_budget", self::n(1536)),
                ("use_joint_pair_polish", b(true)),
            ]);
            t53_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("titan_v6: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("titan_v6");
}
