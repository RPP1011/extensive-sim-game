use crate::ai;

pub fn run_phase6_visualization() {
    let output_path = "generated/reports/ai_phase6_events.html";
    match ai::tooling::export_phase5_event_visualization(output_path, 31, 320) {
        Ok(()) => {
            println!("Phase 6 event visualization written to: {}", output_path);
            println!("Open this file in a browser to explore event filters and timeline.");
        }
        Err(err) => {
            eprintln!("Failed to write phase 6 visualization: {}", err);
        }
    }
}

pub fn run_pathing_visualization() {
    let output_path = "generated/reports/ai_pathing_chokepoint.html";
    match ai::tooling::export_horde_chokepoint_visualization(output_path, 101, 420) {
        Ok(()) => {
            println!("Pathing visualization written to: {}", output_path);
            println!("Open this file in a browser to inspect chokepoints and flows.");
        }
        Err(err) => {
            eprintln!("Failed to write pathing visualization: {}", err);
        }
    }
}

pub fn run_pathing_hero_win_visualization() {
    let output_path = "generated/reports/ai_pathing_chokepoint_hero_win.html";
    match ai::tooling::export_horde_chokepoint_hero_favored_visualization(output_path, 202, 420) {
        Ok(()) => {
            println!(
                "Pathing hero-favored visualization written to: {}",
                output_path
            );
            println!("Open this file in a browser to inspect chokepoints and flows.");
        }
        Err(err) => {
            eprintln!(
                "Failed to write hero-favored pathing visualization: {}",
                err
            );
        }
    }
}

pub fn run_visualization_index() {
    let output_path = "generated/reports/index.html";
    let links = vec![
        (
            "Phase 6 Event Visualization".to_string(),
            "ai_phase6_events.html".to_string(),
        ),
        (
            "Pathing Chokepoint Visualization".to_string(),
            "ai_pathing_chokepoint.html".to_string(),
        ),
        (
            "Custom Scenario Visualization (if generated)".to_string(),
            "ai_custom_scenario.html".to_string(),
        ),
    ];
    match ai::tooling::export_visualization_index(output_path, &links) {
        Ok(()) => {
            println!("Visualization index written to: {}", output_path);
            println!("Open this file in a browser to navigate generated visualizations.");
        }
        Err(err) => {
            eprintln!("Failed to write visualization index: {}", err);
        }
    }
}

pub fn run_write_scenario_template(path: &str) {
    match ai::tooling::write_custom_scenario_template(path) {
        Ok(()) => {
            println!("Scenario template written to: {}", path);
            println!("Edit the json and run --scenario-viz <path> to visualize it.");
        }
        Err(err) => {
            eprintln!("Failed to write scenario template: {}", err);
        }
    }
}

pub fn run_custom_scenario_visualization(scenario_path: &str, output_path: &str) {
    match ai::tooling::export_custom_scenario_visualization(scenario_path, output_path) {
        Ok(()) => {
            println!("Custom scenario visualization written to: {}", output_path);
            println!("Source scenario: {}", scenario_path);
        }
        Err(err) => {
            eprintln!("Failed to write custom scenario visualization: {}", err);
        }
    }
}
