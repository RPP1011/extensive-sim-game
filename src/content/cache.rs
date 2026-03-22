use std::path::{Path, PathBuf};
use serde::{Serialize, de::DeserializeOwned};

/// Filesystem cache for AOT-generated content.
/// Each campaign save gets its own cache directory.
pub struct ContentCache {
    pub cache_dir: PathBuf,
}

impl ContentCache {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self { cache_dir: cache_dir.into() }
    }

    /// Try to load cached content. If not found, generate it, cache it, and return it.
    pub fn load_or_generate<T: Serialize + DeserializeOwned>(
        &self,
        key: &str,
        generate: impl FnOnce() -> T,
    ) -> T {
        let path = self.path_for(key);
        if let Some(cached) = self.try_load::<T>(&path) {
            return cached;
        }
        let value = generate();
        self.try_save(&path, &value);
        value
    }

    /// Load cached content by key.
    pub fn load<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.try_load(&self.path_for(key))
    }

    /// Save content to cache.
    pub fn save<T: Serialize>(&self, key: &str, value: &T) -> bool {
        self.try_save(&self.path_for(key), value)
    }

    /// Invalidate a cached entry.
    pub fn invalidate(&self, key: &str) -> bool {
        let path = self.path_for(key);
        std::fs::remove_file(&path).is_ok()
    }

    /// Clear all cached content.
    pub fn clear(&self) -> bool {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir).is_ok()
        } else {
            true
        }
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{key}.json"))
    }

    fn try_load<T: DeserializeOwned>(&self, path: &Path) -> Option<T> {
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn try_save<T: Serialize>(&self, path: &Path, value: &T) -> bool {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(value) {
            Ok(json) => std::fs::write(path, json).is_ok(),
            Err(_) => false,
        }
    }
}
