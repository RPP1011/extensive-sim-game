//! webband_app — the HOST LAYER of the Webband port: seeded generation
//! (slice S7a) and the campaign brain (slice S7b).
//!
//! S11 adds THE GUILD LAYER — the politics era: the country's powers
//! ([`factions`]), the answer verb ([`petitions`]), the founders' long arc
//! ([`ambition`]), the labour market's clocks ([`bands`]), companies on the
//! road ([`afield`]) and the guild's aging world-knowledge ([`markets`] +
//! [`knowledge`]). It is opt-in per campaign — [`campaign::Campaign::new`]
//! stays exactly as it was and [`campaign::Campaign::new_political`] appends
//! the politics rolls to the seeded stream.
//!
//! Pure logic, no engine/wgpu dependency, ported from `F:\MB` (TypeScript,
//! read-only reference). One call — [`founding::new_founding`] — reproduces
//! Webband's frozen draw order (name → cast → world → goals → colony), then
//! stamps the starting scenario with ZERO draws (the comparability law).
//! Same seed, same founding, forever. Around it, S7b adds the campaign
//! loop's brain as plain functions over plain state: the storyteller
//! ([`director`]), raid math ([`raids`]), the dawn-fold orchestration
//! skeleton and the versioned save root ([`campaign`]) — fixture wiring is
//! a later slice; every fixture-facing effect is a typed event out.
//!
//! Laws carried over from the TS originals:
//! - **Draw-order discipline**: every generation draw comes from one
//!   mulberry32 stream ([`rng::RngState`], seed + counter persisted so a save
//!   can resume it). Never reorder shipped draws.
//! - **Constraint asserts at generation** ([`castgen::assert_cast`]): bad
//!   generation is a loud `Err`, never a quietly wrong save.
//! - **Scenarios are data** ([`scenario::SCENARIOS`]) applied AFTER all rolls
//!   with no rng access at all — the function signature enforces the law.

pub mod afield;
pub mod ambition;
pub mod bands;
pub mod campaign;
pub mod castgen;
pub mod defs;
pub mod director;
pub mod error;
pub mod factions;
pub mod founding;
pub mod knowledge;
pub mod markets;
pub mod noise;
pub mod petitions;
pub mod raids;
pub mod rng;
pub mod scenario;
pub mod worldgen;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_campaign;
#[cfg(test)]
mod tests_politics;
