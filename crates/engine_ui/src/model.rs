/// A value lookup key into UiData (a named sim-readback value).
pub type BindKey = String;

#[derive(Clone, Debug)]
pub enum Widget {
    /// Horizontal bar: value/max fraction, labelled, RGB color.
    Bar {
        label: String,
        value: BindKey,
        max: BindKey,
        color: [u8; 3],
    },
    /// Text with `{key}` placeholders substituted from UiData (formatted as ints).
    Text { template: String },
}

#[derive(Clone, Debug)]
pub enum UiAction {
    /// Increment a named host-side counter (e.g. "bolt_level"). Applied by the caller.
    Increment(String),
    /// Restart the run.
    Restart,
}

#[derive(Clone, Debug)]
pub struct Card {
    pub label: String,
    pub action: UiAction,
}

#[derive(Clone, Debug)]
pub enum Screen {
    /// Modal upgrade menu — pauses; cards are buttons returning their action.
    Menu { title: String, cards: Vec<Card> },
    /// Modal end screen — summary rows (label, BindKey) + a restart button.
    End {
        title: String,
        summary: Vec<(String, BindKey)>,
        restart_label: String,
    },
}

#[derive(Clone, Debug)]
pub struct NamedScreen {
    pub name: String,
    pub screen: Screen,
}

#[derive(Clone, Debug, Default)]
pub struct UiModel {
    pub hud: Vec<Widget>,
    pub screens: Vec<NamedScreen>,
}

impl UiModel {
    pub fn screen(&self, name: &str) -> Option<&Screen> {
        self.screens
            .iter()
            .find(|s| s.name == name)
            .map(|s| &s.screen)
    }
}
