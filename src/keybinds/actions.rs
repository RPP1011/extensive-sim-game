/// All semantic actions the game recognises, independent of physical keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameAction {
    // Navigation
    MenuUp,
    MenuDown,
    MenuLeft,
    MenuRight,
    MenuSelect,
    MenuBack,

    // UI toggles
    ToggleQuestLog,
    ToggleSettings,
    ToggleTutorial,

    // Save / Load
    QuickSave,
    QuickLoad,
    SaveBrowser,
    QuickSaveSlot2,
    QuickLoadSlot2,
    QuickSaveSlot3,
    QuickLoadSlot3,

    // Combat
    PauseResume,
    SpeedUp,
    SlowDown,
    NextUnit,
    PrevUnit,
    ViewBattle,

    // Abilities
    Ability1,
    Ability2,
    Ability3,
    Ability4,
    Ability5,
    Ability6,
    Ability7,
    Ability8,
    Ability9,

    // Camera
    CameraUp,
    CameraDown,
    CameraLeft,
    CameraRight,
    CameraZoomIn,
    CameraZoomOut,

    // Screenshot
    Screenshot,

    // Replay
    ReplayPlayPause,
    ReplayNextFrame,
    ReplayPrevFrame,
    ReplayExit,

    // General
    Confirm,
    Cancel,
}

impl GameAction {
    /// Returns a static slice of every `GameAction` variant (in declaration order).
    pub fn all_variants() -> &'static [GameAction] {
        use GameAction::*;
        &[
            MenuUp,
            MenuDown,
            MenuLeft,
            MenuRight,
            MenuSelect,
            MenuBack,
            ToggleQuestLog,
            ToggleSettings,
            ToggleTutorial,
            QuickSave,
            QuickLoad,
            SaveBrowser,
            QuickSaveSlot2,
            QuickLoadSlot2,
            QuickSaveSlot3,
            QuickLoadSlot3,
            PauseResume,
            SpeedUp,
            SlowDown,
            NextUnit,
            PrevUnit,
            ViewBattle,
            Ability1,
            Ability2,
            Ability3,
            Ability4,
            Ability5,
            Ability6,
            Ability7,
            Ability8,
            Ability9,
            CameraUp,
            CameraDown,
            CameraLeft,
            CameraRight,
            CameraZoomIn,
            CameraZoomOut,
            Screenshot,
            ReplayPlayPause,
            ReplayNextFrame,
            ReplayPrevFrame,
            ReplayExit,
            Confirm,
            Cancel,
        ]
    }

    /// Returns `true` when this action makes sense in the given context.
    pub fn valid_in_context(&self, ctx: InputContext) -> bool {
        use GameAction::*;
        use InputContext::*;
        match self {
            // Navigation actions are valid in menus / character creation / settings
            MenuUp | MenuDown | MenuLeft | MenuRight | MenuSelect | MenuBack => matches!(
                ctx,
                StartMenu | CharacterCreation | Settings | Dialog
            ),

            // UI toggles available during gameplay screens
            ToggleQuestLog => matches!(ctx, Overworld | Combat),
            ToggleSettings => true, // accessible from any context
            ToggleTutorial => matches!(ctx, Overworld | Combat | StartMenu),

            // Save / Load
            QuickSave | QuickLoad | SaveBrowser
            | QuickSaveSlot2 | QuickLoadSlot2
            | QuickSaveSlot3 | QuickLoadSlot3 => matches!(ctx, Overworld | Combat),

            // Combat-specific
            PauseResume | SpeedUp | SlowDown | NextUnit | PrevUnit | ViewBattle => {
                matches!(ctx, Combat)
            }

            // Abilities only in combat
            Ability1 | Ability2 | Ability3 | Ability4 | Ability5 | Ability6 | Ability7
            | Ability8 | Ability9 => matches!(ctx, Combat),

            // Camera movement in spatial views
            CameraUp | CameraDown | CameraLeft | CameraRight | CameraZoomIn
            | CameraZoomOut => matches!(ctx, Overworld | Combat | Replay),

            // Screenshot is always available
            Screenshot => true,

            // Replay-specific
            ReplayPlayPause | ReplayNextFrame | ReplayPrevFrame | ReplayExit => {
                matches!(ctx, Replay)
            }

            // General
            Confirm => true,
            Cancel => true,
        }
    }
}

/// The current UI / gameplay context, used to filter which actions are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputContext {
    StartMenu,
    CharacterCreation,
    Overworld,
    Combat,
    Replay,
    Dialog,
    Settings,
}
