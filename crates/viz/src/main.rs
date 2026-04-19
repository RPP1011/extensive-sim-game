mod app;
mod state;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <scenario.toml>", args.first().map(String::as_str).unwrap_or("viz"));
        std::process::exit(2);
    }
    let path = std::path::PathBuf::from(&args[1]);
    let scenario = viz::scenario::load(&path)?;
    app::run(scenario, path)
}
