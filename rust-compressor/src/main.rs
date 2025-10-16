use std::{
    env,
    fs::File,
    io::{Read, Write},
};

use rolling_hash_rs::compressor;

fn main() -> std::io::Result<()> {
    // parse args
    let args: Vec<String> = env::args().collect();
    let in_name = &args[1];
    let out_name = &args[2];
    let mut text = String::new();
    File::open(in_name)?.read_to_string(&mut text).unwrap();

    let start = std::time::Instant::now();
    let (_replacements, compressed) = compressor::greedy(text).unwrap();
    println!("Greedy in {}", start.elapsed().as_millis());

    let mut out_file = File::create(out_name)?;
    let contents: Vec<u8> = compressed.bytes().into_iter().collect();
    out_file.write_all(&contents)?;
    Ok(())
}
