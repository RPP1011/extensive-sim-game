use bevy_egui::egui;

pub fn lerp_channel(a: u8, b: u8, t: f32) -> u8 {
    let t = t.clamp(0.0, 1.0);
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

pub fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    egui::Color32::from_rgb(
        lerp_channel(a.r(), b.r(), t),
        lerp_channel(a.g(), b.g(), t),
        lerp_channel(a.b(), b.b(), t),
    )
}

pub fn paint_landscape_backsplash(ui: &egui::Ui, cinematic_tint: bool) {
    let rect = ui.max_rect();
    let painter = ui.painter_at(rect);
    let sky_top = if cinematic_tint {
        egui::Color32::from_rgb(16, 58, 86)
    } else {
        egui::Color32::from_rgb(20, 70, 104)
    };
    let sky_bottom = if cinematic_tint {
        egui::Color32::from_rgb(8, 20, 34)
    } else {
        egui::Color32::from_rgb(10, 24, 40)
    };
    let gradient_steps = 48;
    for idx in 0..gradient_steps {
        let t0 = idx as f32 / gradient_steps as f32;
        let t1 = (idx + 1) as f32 / gradient_steps as f32;
        let y0 = rect.top() + rect.height() * t0;
        let y1 = rect.top() + rect.height() * t1;
        let band =
            egui::Rect::from_min_max(egui::pos2(rect.left(), y0), egui::pos2(rect.right(), y1));
        painter.rect_filled(band, 0.0, lerp_color(sky_top, sky_bottom, t0));
    }

    let sun_center = egui::pos2(
        rect.right() - rect.width() * 0.2,
        rect.top() + rect.height() * 0.22,
    );
    painter.circle_filled(
        sun_center,
        rect.width() * 0.11,
        egui::Color32::from_rgba_premultiplied(246, 212, 133, 46),
    );
    painter.circle_filled(
        sun_center,
        rect.width() * 0.06,
        egui::Color32::from_rgba_premultiplied(248, 230, 171, 90),
    );

    let ridge_a = vec![
        egui::pos2(rect.left() - 40.0, rect.bottom() - rect.height() * 0.24),
        egui::pos2(rect.left() + rect.width() * 0.18, rect.bottom() - rect.height() * 0.58),
        egui::pos2(rect.left() + rect.width() * 0.38, rect.bottom() - rect.height() * 0.32),
        egui::pos2(rect.left() + rect.width() * 0.56, rect.bottom() - rect.height() * 0.61),
        egui::pos2(rect.left() + rect.width() * 0.82, rect.bottom() - rect.height() * 0.34),
        egui::pos2(rect.right() + 40.0, rect.bottom() - rect.height() * 0.5),
        egui::pos2(rect.right() + 40.0, rect.bottom() + 2.0),
        egui::pos2(rect.left() - 40.0, rect.bottom() + 2.0),
    ];
    painter.add(egui::Shape::convex_polygon(
        ridge_a,
        egui::Color32::from_rgba_premultiplied(17, 30, 45, 210),
        egui::Stroke::NONE,
    ));

    let ridge_b = vec![
        egui::pos2(rect.left() - 40.0, rect.bottom() - rect.height() * 0.14),
        egui::pos2(rect.left() + rect.width() * 0.16, rect.bottom() - rect.height() * 0.32),
        egui::pos2(rect.left() + rect.width() * 0.31, rect.bottom() - rect.height() * 0.18),
        egui::pos2(rect.left() + rect.width() * 0.49, rect.bottom() - rect.height() * 0.34),
        egui::pos2(rect.left() + rect.width() * 0.68, rect.bottom() - rect.height() * 0.19),
        egui::pos2(rect.right() + 40.0, rect.bottom() - rect.height() * 0.3),
        egui::pos2(rect.right() + 40.0, rect.bottom() + 2.0),
        egui::pos2(rect.left() - 40.0, rect.bottom() + 2.0),
    ];
    painter.add(egui::Shape::convex_polygon(
        ridge_b,
        egui::Color32::from_rgba_premultiplied(11, 21, 31, 232),
        egui::Stroke::NONE,
    ));

    let horizon = egui::Rect::from_min_max(
        egui::pos2(rect.left(), rect.bottom() - rect.height() * 0.14),
        egui::pos2(rect.right(), rect.bottom()),
    );
    painter.rect_filled(
        horizon,
        0.0,
        egui::Color32::from_rgba_premultiplied(7, 14, 20, 242),
    );
}

pub fn illustration_placeholder_tile(ui: &mut egui::Ui, title: &str, caption: &str, color: egui::Color32) {
    egui::Frame::none()
        .fill(egui::Color32::from_rgba_premultiplied(9, 13, 18, 210))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_premultiplied(118, 156, 182, 96),
        ))
        .rounding(egui::Rounding::same(8.0))
        .inner_margin(egui::Margin::same(10.0))
        .show(ui, |ui| {
            ui.colored_label(color, title);
            let width = ui.available_width().max(160.0);
            let (response, painter) =
                ui.allocate_painter(egui::vec2(width, 90.0), egui::Sense::hover());
            let rect = response.rect;
            painter.rect_filled(
                rect,
                6.0,
                egui::Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), 38),
            );
            painter.rect_stroke(
                rect.shrink(2.0),
                5.0,
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_premultiplied(204, 224, 235, 44),
                ),
            );
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Illustration",
                egui::FontId::proportional(15.0),
                egui::Color32::from_rgba_premultiplied(224, 236, 242, 190),
            );
            ui.small(caption);
        });
}

pub fn split_faction_impact_sections(impact: &str) -> (String, String, String) {
    let mut doctrine = impact.trim().to_string();
    let mut profile = "Starting profile unavailable.".to_string();
    let mut recruit = "Recruit pool bias unavailable.".to_string();

    if let Some((left, right)) = impact.split_once("Starting profile:") {
        doctrine = left.trim().trim_end_matches('.').to_string();
        let remainder = right.trim();
        if let Some((profile_section, recruit_section)) = remainder.split_once("Recruit pools") {
            profile = format!("Starting profile: {}", profile_section.trim().trim_end_matches('.'));
            recruit = format!("Recruit pools{}", recruit_section.trim());
        } else if !remainder.is_empty() {
            profile = format!("Starting profile: {}", remainder.trim().trim_end_matches('.'));
        }
    }

    (doctrine, profile, recruit)
}
