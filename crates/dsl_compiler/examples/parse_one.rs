fn main() {
    let name = std::env::args().nth(1).expect("usage: parse_one <fixture>");
    let path = format!("assets/sim/{name}.sim");
    let src = std::fs::read_to_string(&path).expect("read .sim");
    match dsl_compiler::parse(&src) {
        Ok(_) => println!("{name}: OK"),
        Err(e) => println!("{name}: FAIL ({:?})", e.message),
    }
}
