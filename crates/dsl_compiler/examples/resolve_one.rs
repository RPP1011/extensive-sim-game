fn main() {
    let name = std::env::args().nth(1).expect("usage: resolve_one <fixture>");
    let path = format!("assets/sim/{name}.sim");
    let src = std::fs::read_to_string(&path).expect("read .sim");
    let parsed = match dsl_compiler::parse(&src) {
        Ok(p) => p,
        Err(e) => {
            println!("{name}: PARSE FAIL\n{}", e.rendered);
            return;
        }
    };
    match dsl_compiler::resolve::resolve(parsed) {
        Ok(_) => println!("{name}: RESOLVE OK"),
        Err(e) => println!("{name}: RESOLVE FAIL\n{e:?}"),
    }
}
