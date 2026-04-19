//! Print the current schema hash as hex. Useful for regenerating the
//! `.schema_hash` baseline after an intentional schema change:
//!
//! ```bash
//! cargo run -p engine --example print_schema_hash > crates/engine/.schema_hash
//! ```

fn main() {
    let h = engine::schema_hash::schema_hash();
    for b in &h {
        print!("{:02x}", b);
    }
    println!();
}
