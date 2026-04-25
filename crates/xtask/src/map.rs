use std::process::ExitCode;

use super::cli::MapVoronoiArgs;

pub fn run_map_voronoi(_args: MapVoronoiArgs) -> ExitCode {
    eprintln!("map voronoi command is deprecated — overworld grid has been removed");
    ExitCode::from(1)
}
