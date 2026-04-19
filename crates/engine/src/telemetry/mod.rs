pub mod metrics;
pub mod sink;
pub mod sinks;

pub use sink::TelemetrySink;
pub use sinks::{FileSink, NullSink, VecSink};
