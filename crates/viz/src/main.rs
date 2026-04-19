mod app;
mod state;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <scenario.toml>", args.first().map(String::as_str).unwrap_or("viz"));
        std::process::exit(2);
    }
    let scenario = viz::scenario::load(&args[1])?;
    app::run(scenario)
}
