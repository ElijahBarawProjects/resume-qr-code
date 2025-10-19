use std::{
    env,
    fs::File,
    io::{Read, Write},
};

use rolling_hash_rs::compressor;

fn _main() -> std::io::Result<()> {
    // parse args
    let args: Vec<String> = env::args().collect();
    let in_name = &args[1];
    let out_name = &args[2];
    let mut text = String::new();
    File::open(in_name)?.read_to_string(&mut text).unwrap();
    // strip head
    let text = text
        .strip_prefix("<!doctypehtml><meta charset=\"utf-8\">")
        .and_then(|s| Some(s.to_string()))
        .unwrap_or(text);

    let start = std::time::Instant::now();
    let compressed = compressor::compress(text).unwrap();
    println!("Greedy in {}", start.elapsed().as_millis());

    let mut out_file = File::create(out_name)?;
    let contents: Vec<u8> = compressed.bytes().into_iter().collect();
    out_file.write_all(&contents)?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    _main()
}
