//! Per-system timing accumulator for the `profile-systems` feature.
//!
//! When the feature is enabled, each call to a system is timed and recorded
//! to a thread-local accumulator. At end of tick, accumulators from all
//! threads are folded into a `Vec<SystemTiming>` on the `TickProfile`.
//!
//! Zero overhead when the feature is disabled: `SystemTiming` is a stub
//! struct, `record()` and `drain()` are no-ops.

#[derive(Debug, Clone, Default)]
pub struct SystemTiming {
    pub name: &'static str,
    pub total_ns: u64,
    pub calls: u32,
    pub entities_touched: u64,
}

#[cfg(feature = "profile-systems")]
pub use enabled::*;

#[cfg(not(feature = "profile-systems"))]
pub use disabled::*;

#[cfg(feature = "profile-systems")]
mod enabled {
    use super::SystemTiming;
    use std::cell::RefCell;
    use std::collections::HashMap;

    #[derive(Debug, Default)]
    pub struct SystemProfileAccumulator {
        map: HashMap<&'static str, (u64, u32, u64)>,
    }

    impl SystemProfileAccumulator {
        pub fn record(&mut self, name: &'static str, ns: u64, touched: u32) {
            let entry = self.map.entry(name).or_insert((0, 0, 0));
            entry.0 += ns;
            entry.1 += 1;
            entry.2 += touched as u64;
        }

        pub fn merge(&mut self, other: Self) {
            for (k, (ns, calls, touched)) in other.map {
                let entry = self.map.entry(k).or_insert((0, 0, 0));
                entry.0 += ns;
                entry.1 += calls;
                entry.2 += touched;
            }
        }

        pub fn into_timings(self) -> Vec<SystemTiming> {
            self.map
                .into_iter()
                .map(|(name, (total_ns, calls, entities_touched))| SystemTiming {
                    name,
                    total_ns,
                    calls,
                    entities_touched,
                })
                .collect()
        }

        pub fn is_empty(&self) -> bool { self.map.is_empty() }
    }

    thread_local! {
        static THREAD_ACC: RefCell<SystemProfileAccumulator> =
            RefCell::new(SystemProfileAccumulator::default());
    }

    pub fn thread_record(name: &'static str, ns: u64, touched: u32) {
        THREAD_ACC.with(|a| a.borrow_mut().record(name, ns, touched));
    }

    pub fn thread_drain() -> SystemProfileAccumulator {
        THREAD_ACC.with(|a| std::mem::take(&mut *a.borrow_mut()))
    }
}

#[cfg(not(feature = "profile-systems"))]
mod disabled {
    #[derive(Debug, Default)]
    pub struct SystemProfileAccumulator;

    impl SystemProfileAccumulator {
        pub fn record(&mut self, _name: &'static str, _ns: u64, _touched: u32) {}
        pub fn merge(&mut self, _other: Self) {}
        pub fn into_timings(self) -> Vec<super::SystemTiming> { Vec::new() }
        pub fn is_empty(&self) -> bool { true }
    }

    #[inline(always)]
    pub fn thread_record(_name: &'static str, _ns: u64, _touched: u32) {}

    #[inline(always)]
    pub fn thread_drain() -> SystemProfileAccumulator { SystemProfileAccumulator }
}

#[cfg(test)]
#[cfg(feature = "profile-systems")]
mod tests {
    use super::*;

    #[test]
    fn accumulator_records_and_folds() {
        let mut acc = SystemProfileAccumulator::default();
        acc.record("foo", 1000, 10);
        acc.record("foo", 2000, 20);
        acc.record("bar", 500, 5);
        let timings = acc.into_timings();
        let foo = timings.iter().find(|t| t.name == "foo").unwrap();
        assert_eq!(foo.total_ns, 3000);
        assert_eq!(foo.calls, 2);
        assert_eq!(foo.entities_touched, 30);
        let bar = timings.iter().find(|t| t.name == "bar").unwrap();
        assert_eq!(bar.total_ns, 500);
    }

    #[test]
    fn thread_local_record_and_drain() {
        // Clear any prior state
        let _ = thread_drain();
        thread_record("x", 100, 1);
        thread_record("x", 200, 2);
        let acc = thread_drain();
        let timings = acc.into_timings();
        let x = timings.iter().find(|t| t.name == "x").unwrap();
        assert_eq!(x.total_ns, 300);
        assert_eq!(x.calls, 2);
        assert_eq!(x.entities_touched, 3);
    }
}
