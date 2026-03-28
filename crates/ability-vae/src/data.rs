//! Dataset loader: reads .ability files, tokenizes them, produces batches.

use burn::prelude::*;
use std::path::Path;
use tactical_sim::sim::ability_transformer::tokenizer::AbilityTokenizer;

/// Tokenized ability dataset — all sequences padded to max_len.
pub struct AbilityDataset {
    /// All token sequences, each padded to max_len. [N, max_len]
    sequences: Vec<Vec<u32>>,
    /// Length of each sequence (before padding).
    lengths: Vec<usize>,
    /// Vocabulary size.
    vocab_size: usize,
    /// Max sequence length.
    max_len: usize,
    /// Pad token ID.
    pad_id: u32,
}

impl AbilityDataset {
    /// Load all .ability files from a directory tree, tokenize, and build dataset.
    pub fn load<B: Backend>(dir: &str, _device: &B::Device) -> Self {
        let tokenizer = AbilityTokenizer::new();
        let vocab_size = tokenizer.vocab_size();
        let pad_id = 0u32; // [PAD] is always ID 0

        let mut all_sequences = Vec::new();

        // Recursively find .ability files
        let ability_files = find_ability_files(Path::new(dir));
        println!("  Found {} .ability files", ability_files.len());

        for path in &ability_files {
            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Split file into individual ability blocks
            let blocks = split_ability_blocks(&content);
            for block in blocks {
                let tokens = tokenizer.encode_with_cls(&block);
                if tokens.len() >= 4 {
                    // Skip trivially short sequences
                    all_sequences.push(tokens);
                }
            }
        }

        println!("  Tokenized {} abilities", all_sequences.len());

        // Determine max length (cap at 128)
        let max_len = all_sequences.iter().map(|s| s.len()).max().unwrap_or(1).min(128);
        println!("  Max sequence length: {} (capped at {})",
            all_sequences.iter().map(|s| s.len()).max().unwrap_or(0), max_len);

        // Pad/truncate all sequences
        let mut lengths = Vec::with_capacity(all_sequences.len());
        for seq in &mut all_sequences {
            let orig_len = seq.len().min(max_len);
            lengths.push(orig_len);
            seq.truncate(max_len);
            while seq.len() < max_len {
                seq.push(pad_id);
            }
        }

        // Shuffle deterministically
        let mut rng_state: u64 = 42;
        for i in (1..all_sequences.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state >> 33) as usize % (i + 1);
            all_sequences.swap(i, j);
            lengths.swap(i, j);
        }

        Self {
            sequences: all_sequences,
            lengths,
            vocab_size,
            max_len,
            pad_id,
        }
    }

    pub fn num_samples(&self) -> usize {
        self.sequences.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get raw token sequences for a range of indices.
    pub fn get_sequences(&self, range: std::ops::Range<usize>) -> Vec<&[u32]> {
        range.filter_map(|i| self.sequences.get(i).map(|s| s.as_slice())).collect()
    }

    /// Generate batches of (input_ids, target_ids, mask) for a range of indices.
    ///
    /// - input_ids: the full token sequence [B, T]
    /// - target_ids: same as input_ids (for teacher-forced reconstruction)
    /// - mask: bool tensor, true where token is non-pad [B, T]
    pub fn batches<B: Backend>(
        &self,
        range: std::ops::Range<usize>,
        batch_size: usize,
        device: &B::Device,
    ) -> Vec<(Tensor<B, 2, Int>, Tensor<B, 2, Int>, Tensor<B, 2, Bool>)> {
        let indices: Vec<usize> = range.collect();
        let mut batches = Vec::new();

        for chunk in indices.chunks(batch_size) {
            let b = chunk.len();
            let t = self.max_len;

            let mut input_data = vec![0i64; b * t];
            let mut mask_data = vec![false; b * t];

            for (bi, &idx) in chunk.iter().enumerate() {
                let seq = &self.sequences[idx];
                let len = self.lengths[idx];
                for ti in 0..t {
                    input_data[bi * t + ti] = seq[ti] as i64;
                    mask_data[bi * t + ti] = ti < len;
                }
            }

            let input_ids = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(input_data.clone(), [b * t]),
                device,
            ).reshape([b, t]);

            let target_ids = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(input_data, [b * t]),
                device,
            ).reshape([b, t]);

            let mask = Tensor::<B, 1, Bool>::from_data(
                burn::tensor::TensorData::new(mask_data, [b * t]),
                device,
            ).reshape([b, t]);

            batches.push((input_ids, target_ids, mask));
        }

        batches
    }
}

/// Recursively find all .ability files under a directory.
fn find_ability_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if !dir.exists() {
        return files;
    }
    if dir.is_file() && dir.extension().is_some_and(|e| e == "ability") {
        files.push(dir.to_path_buf());
        return files;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_ability_files(&path));
            } else if path.extension().is_some_and(|e| e == "ability") {
                files.push(path);
            }
        }
    }
    files
}

/// Split a file containing multiple ability/passive blocks into individual blocks.
fn split_ability_blocks(content: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut brace_depth = 0i32;
    let mut in_block = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Skip pure comment lines outside blocks
        if !in_block && (trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.is_empty()) {
            continue;
        }

        // Start of a new block
        if !in_block && (trimmed.starts_with("ability ") || trimmed.starts_with("passive ")) {
            in_block = true;
            current.clear();
        }

        if in_block {
            current.push_str(line);
            current.push('\n');

            for ch in trimmed.chars() {
                match ch {
                    '{' => brace_depth += 1,
                    '}' => brace_depth -= 1,
                    _ => {}
                }
            }

            if brace_depth <= 0 && current.contains('{') {
                blocks.push(current.trim().to_string());
                current.clear();
                brace_depth = 0;
                in_block = false;
            }
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_blocks() {
        let input = r#"
// Some comment
ability Fireball {
    target: enemy, range: 5.0
    cooldown: 5s
    hint: damage

    damage 50 [FIRE: 60]
}

ability Heal {
    target: ally, range: 4.0
    cooldown: 8s
    hint: heal

    heal 40
}
"#;
        let blocks = split_ability_blocks(input);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].starts_with("ability Fireball"));
        assert!(blocks[1].starts_with("ability Heal"));
    }
}
