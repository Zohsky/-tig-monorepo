use anyhow::Result;
use clap::{arg, Command};
#[cfg(any(
    feature = "c001",
    feature = "c002",
    feature = "c003",
    feature = "c004",
    feature = "c005",
    feature = "c006",
    feature = "c007",
    feature = "c008"
))]
use serde_json::{Map, Value};
#[cfg(any(
    feature = "c001",
    feature = "c002",
    feature = "c003",
    feature = "c004",
    feature = "c005",
    feature = "c006",
    feature = "c007",
    feature = "c008"
))]
use std::io::Read;
use std::{fs, path::PathBuf};
#[cfg(any(
    feature = "c001",
    feature = "c002",
    feature = "c003",
    feature = "c004",
    feature = "c005",
    feature = "c006",
    feature = "c007",
    feature = "c008"
))]
use tig_challenges::*;
use tig_structs::core::BenchmarkSettings;
use tig_utils::dejsonify;

#[cfg(feature = "cuda")]
use cudarc::{driver::CudaContext, nvrtc::Ptx, runtime::result::device::get_device_prop};

fn cli() -> Command {
    Command::new("tig-verifier")
        .about("Verifies a solution")
        .arg_required_else_help(true)
        .arg(
            arg!(<SETTINGS> "Settings json string or path to json file")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(
            arg!(<RAND_HASH> "A string used in seed generation")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(arg!(<NONCE> "Nonce value").value_parser(clap::value_parser!(u64)))
        .arg(
            arg!(<SOLUTION> "Solution base64 string, path to json file with solution field, or '-' for stdin")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(arg!(--ptx [PTX] "Path to a CUDA ptx file").value_parser(clap::value_parser!(PathBuf)))
        .arg(arg!(--gpu [GPU] "Which GPU device to use").value_parser(clap::value_parser!(usize)))
        .arg(arg!(--verbose "Enable verbose output").action(clap::ArgAction::SetTrue))
}

fn main() {
    let matches = cli().get_matches();

    if let Err(e) = verify_solution(
        matches.get_one::<String>("SETTINGS").unwrap().clone(),
        matches.get_one::<String>("RAND_HASH").unwrap().clone(),
        *matches.get_one::<u64>("NONCE").unwrap(),
        matches.get_one::<String>("SOLUTION").unwrap().clone(),
        matches.get_one::<PathBuf>("ptx").cloned(),
        matches.get_one::<usize>("gpu").cloned(),
        matches.get_one::<bool>("verbose").cloned().unwrap_or(false),
    ) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

pub fn verify_solution(
    settings: String,
    rand_hash: String,
    nonce: u64,
    solution_path: String,
    ptx_path: Option<PathBuf>,
    gpu_device: Option<usize>,
    verbose: bool,
) -> Result<()> {
    let settings = load_settings(&settings);
    let seed = settings.calc_seed(&rand_hash, nonce);

    #[allow(unused_mut)] // Mutated by feature-specific dispatch expansions.
    let mut err_msg = Option::<String>::None;
    // These values are consumed by feature-specific macro expansions. Keep the
    // featureless CLI build warning-free while still returning a useful error.
    let _ = (
        &seed,
        &solution_path,
        &ptx_path,
        &gpu_device,
        verbose,
        &err_msg,
    );

    #[allow(unused_macros)]
    macro_rules! dispatch_challenge {
        ($c:ident, cpu) => {{
            let track_id = if settings.track_id.starts_with('"') && settings.track_id.ends_with('"')
            {
                settings.track_id.clone()
            } else {
                format!(r#""{}""#, settings.track_id)
            };
            let track = serde_json::from_str(&track_id).map_err(|_| {
                anyhow::anyhow!(
                    "Failed to parse track_id '{}' as {}::Track",
                    settings.track_id,
                    stringify!($c)
                )
            })?;
            let challenge = $c::Challenge::generate_instance(&seed, &track).unwrap();
            if verbose {
                println!("{:?}", challenge);
            }

            let solution = load_solution(&solution_path);
            match serde_json::from_str::<$c::Solution>(&solution) {
                Ok(solution) => {
                    if verbose {
                        println!("{:?}", solution);
                    }
                    match challenge.evaluate_solution(&solution) {
                        Ok(quality) => println!("quality: {}", quality),
                        Err(e) => err_msg = Some(format!("Invalid solution: {}", e)),
                    }
                }
                Err(_) => {
                    err_msg = Some(format!(
                        "Invalid solution. Cannot convert to {}::Solution",
                        stringify!($c)
                    ))
                }
            }
        }};

        ($c:ident, gpu) => {{
            let track_id = if settings.track_id.starts_with('"') && settings.track_id.ends_with('"')
            {
                settings.track_id.clone()
            } else {
                format!(r#""{}""#, settings.track_id)
            };
            let track = serde_json::from_str(&track_id).map_err(|_| {
                anyhow::anyhow!(
                    "Failed to parse track_id '{}' as {}::Track",
                    settings.track_id,
                    stringify!($c)
                )
            })?;

            if ptx_path.is_none() {
                panic!("PTX file is required for GPU challenges.");
            }

            let num_gpus = CudaContext::device_count()?;
            if num_gpus == 0 {
                panic!("No CUDA devices found");
            }
            let gpu_device = gpu_device.unwrap_or((nonce % num_gpus as u64) as usize);
            let ptx = Ptx::from_file(ptx_path.unwrap());
            let ctx = CudaContext::new(gpu_device).unwrap();
            ctx.set_blocking_synchronize()?;
            let module = ctx.load_module(ptx).unwrap();
            let stream = ctx.default_stream();
            let prop = get_device_prop(gpu_device as i32).unwrap();

            let challenge = $c::Challenge::generate_instance(
                &seed,
                &track,
                module.clone(),
                stream.clone(),
                &prop,
            )
            .unwrap();

            let solution = load_solution(&solution_path);
            match serde_json::from_str::<$c::Solution>(&solution) {
                Ok(solution) => {
                    if verbose {
                        println!("{:?}", solution);
                    }
                    match challenge.evaluate_solution(
                        &solution,
                        module.clone(),
                        stream.clone(),
                        &prop,
                    ) {
                        Ok(quality) => {
                            stream.synchronize()?;
                            ctx.synchronize()?;
                            println!("quality: {}", quality);
                        }
                        Err(e) => err_msg = Some(format!("Invalid solution: {}", e)),
                    }
                }
                Err(_) => {
                    err_msg = Some(format!(
                        "Invalid solution. Cannot convert to {}::Solution",
                        stringify!($c)
                    ))
                }
            }
        }};
    }

    match settings.challenge_id.as_str() {
        "c001" => {
            #[cfg(not(feature = "c001"))]
            {
                Err(anyhow::anyhow!(
                    "tig-verifier was not compiled with '--features c001'"
                ))
            }
            #[cfg(feature = "c001")]
            {
                dispatch_challenge!(c001, cpu);
                Ok(())
            }
        }
        "c002" => {
            #[cfg(not(feature = "c002"))]
            {
                Err(anyhow::anyhow!(
                    "tig-verifier was not compiled with '--features c002'"
                ))
            }
            #[cfg(feature = "c002")]
            {
                dispatch_challenge!(c002, cpu);
                Ok(())
            }
        }
        "c003" => {
            #[cfg(not(feature = "c003"))]
            {
                Err(anyhow::anyhow!(
                    "tig-verifier was not compiled with '--features c003'"
                ))
            }
            #[cfg(feature = "c003")]
            {
                dispatch_challenge!(c003, cpu);
                Ok(())
            }
        }
        "c004" => {
            #[cfg(not(feature = "c004"))]
            {
                Err(anyhow::anyhow!(
                    "tig-verifier was not compiled with '--features c004'"
                ))
            }
            #[cfg(feature = "c004")]
            {
                dispatch_challenge!(c004, gpu);
                Ok(())
            }
        }
        "c005" => {
            #[cfg(not(feature = "c005"))]
            {
                Err(anyhow::anyhow!(
                    "tig-verifier was not compiled with '--features c005'"
                ))
            }
            #[cfg(feature = "c005")]
            {
                dispatch_challenge!(c005, gpu);
                Ok(())
            }
        }
        "c006" => {
            #[cfg(not(feature = "c006"))]
            {
                Err(anyhow::anyhow!(
                    "tig-verifier was not compiled with '--features c006'"
                ))
            }
            #[cfg(feature = "c006")]
            {
                dispatch_challenge!(c006, gpu);
                Ok(())
            }
        }
        "c007" => {
            #[cfg(not(feature = "c007"))]
            {
                Err(anyhow::anyhow!(
                    "tig-verifier was not compiled with '--features c007'"
                ))
            }
            #[cfg(feature = "c007")]
            {
                dispatch_challenge!(c007, cpu);
                Ok(())
            }
        }
        "c008" => {
            #[cfg(not(feature = "c008"))]
            {
                Err(anyhow::anyhow!(
                    "tig-verifier was not compiled with '--features c008'"
                ))
            }
            #[cfg(feature = "c008")]
            {
                dispatch_challenge!(c008, cpu);
                Ok(())
            }
        }
        _ => Err(anyhow::anyhow!(
            "Unsupported challenge: {}",
            settings.challenge_id
        )),
    }?;

    if let Some(err_msg) = err_msg {
        eprintln!("Verification error: {}", err_msg);
        std::process::exit(1);
    }

    Ok(())
}

fn load_settings(settings: &str) -> BenchmarkSettings {
    let settings = if settings.ends_with(".json") {
        fs::read_to_string(settings).unwrap_or_else(|_| {
            eprintln!("Failed to read settings file: {}", settings);
            std::process::exit(1);
        })
    } else {
        settings.to_string()
    };

    dejsonify::<BenchmarkSettings>(&settings).unwrap_or_else(|_| {
        eprintln!("Failed to parse settings");
        std::process::exit(1);
    })
}

#[cfg(any(
    feature = "c001",
    feature = "c002",
    feature = "c003",
    feature = "c004",
    feature = "c005",
    feature = "c006",
    feature = "c007",
    feature = "c008"
))]
fn load_solution(solution: &str) -> String {
    if solution == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .unwrap_or_else(|_| {
                eprintln!("Failed to read solution from stdin");
                std::process::exit(1);
            });
        buffer
    } else if solution.ends_with(".json") {
        let d = fs::read_to_string(&solution).unwrap_or_else(|_| {
            eprintln!("Failed to read solution file: {}", solution);
            std::process::exit(1);
        });
        let d = serde_json::from_str::<Map<String, Value>>(&d).unwrap_or_else(|_| {
            eprintln!("Failed to parse solution file: {}", solution);
            std::process::exit(1);
        });
        match d.get("solution") {
            None => {
                eprintln!("json file does not contain 'solution' field: {}", solution);
                std::process::exit(1);
            }
            Some(v) => match v.as_str() {
                None => {
                    eprintln!("invalid 'solution' field in json file. Expecting string");
                    std::process::exit(1);
                }
                Some(s) => s.to_string(),
            },
        }
    } else {
        solution.to_string()
    }
}
