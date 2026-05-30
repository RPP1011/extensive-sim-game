use crate::{Screen, UiAction, UiData, UiModel, Widget};

/// Draw the HUD + optional modal screen. Returns the action the user
/// triggered this frame (card click / restart), if any. Caller applies it.
pub fn draw(
    ctx: &egui::Context,
    model: &UiModel,
    data: &UiData,
    active: Option<&str>,
) -> Option<UiAction> {
    // HUD — top-left, non-interactive overlay.
    egui::Area::new(egui::Id::new("engine_ui_hud"))
        .fixed_pos(egui::pos2(12.0, 12.0))
        .show(ctx, |ui| {
            for w in &model.hud {
                match w {
                    Widget::Text { template } => {
                        ui.label(data.fill(template));
                    }
                    Widget::Bar {
                        label,
                        value,
                        max,
                        color,
                    } => {
                        let v = data.get(value);
                        let m = data.get(max).max(1e-3);
                        let frac = (v / m).clamp(0.0, 1.0);
                        let col = egui::Color32::from_rgb(color[0], color[1], color[2]);
                        ui.horizontal(|ui| {
                            ui.label(label);
                            ui.add(egui::ProgressBar::new(frac).fill(col).desired_width(180.0));
                        });
                    }
                }
            }
        });

    // Modal screen.
    let mut action = None;
    if let Some(name) = active {
        if let Some(screen) = model.screen(name) {
            egui::Window::new("engine_ui_modal")
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| match screen {
                    Screen::Menu { title, cards } => {
                        ui.heading(title);
                        for c in cards {
                            if ui.button(&c.label).clicked() {
                                action = Some(c.action.clone());
                            }
                        }
                    }
                    Screen::End {
                        title,
                        summary,
                        restart_label,
                    } => {
                        ui.heading(title);
                        for (label, key) in summary {
                            ui.label(format!("{label}: {}", data.fill(&format!("{{{key}}}"))));
                        }
                        if ui.button(restart_label).clicked() {
                            action = Some(UiAction::Restart);
                        }
                    }
                });
        }
    }
    action
}

#[cfg(test)]
mod tests {
    #[test]
    fn draw_hud_headless_no_panic() {
        use crate::*;
        let model = UiModel {
            hud: vec![
                Widget::Bar {
                    label: "HP".into(),
                    value: "hp".into(),
                    max: "hp_max".into(),
                    color: [220, 40, 40],
                },
                Widget::Text {
                    template: "Lv {level}  Kills {kills}".into(),
                },
            ],
            screens: vec![NamedScreen {
                name: "level_up".into(),
                screen: Screen::Menu {
                    title: "Level Up".into(),
                    cards: vec![Card {
                        label: "Bolt +".into(),
                        action: UiAction::Increment("bolt_level".into()),
                    }],
                },
            }],
        };
        let mut data = UiData::new();
        data.set("hp", 50.0)
            .set("hp_max", 100.0)
            .set("level", 2.0)
            .set("kills", 7.0);
        let ctx = egui::Context::default();
        let out = ctx.run(egui::RawInput::default(), |ctx| {
            let _ = render::draw(ctx, &model, &data, Some("level_up"));
        });
        assert!(!out.shapes.is_empty(), "draw produced no shapes");
    }
}
