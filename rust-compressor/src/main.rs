use clap::Parser;
use regex::Regex;
use std::{
    fs::File,
    io::{self, Read, Write},
};

use rolling_hash_rs::compressor;

/// HTML compression tool using greedy substring replacement algorithm
#[derive(Parser, Debug)]
#[command(
    name = "html-compressor",
    about = "Process input from a file or stdin and output to a file or stdout",
    long_about = "Compresses HTML content using a greedy algorithm to find and replace\nrepeated substrings with shorter keys, generating self-expanding HTML."
)]
struct Args {
    /// Path to the input HTML file
    #[arg(long, group = "input")]
    input_file: Option<String>,

    /// Read input from standard input
    #[arg(long, group = "input")]
    stdin: bool,

    /// Path to the output HTML file
    #[arg(long, group = "output")]
    output_file: Option<String>,

    /// Write output to standard output
    #[arg(long, group = "output")]
    stdout: bool,

    /// Enable verbose output showing replacement details
    #[arg(short, long)]
    verbose: bool,

    /// Enable hash collision detection and reporting
    #[arg(long, default_value_t = false)]
    check_collisions: bool,

    /// Minimum substring length to consider (default: 3)
    #[arg(long, default_value_t = 3)]
    min_len: usize,

    /// Maximum substring length to consider (default: 50)
    #[arg(long, default_value_t = 50)]
    max_len: usize,

    /// Allow overlapping substrings when counting occurrences
    #[arg(long, default_value_t = true)]
    allow_overlaps: bool,

    /// Number of threads to use for parallel processing (default: number of CPUs)
    #[arg(long)]
    nproc: Option<usize>,
}

use crate::compressor::CompressorConfig;

impl TryFrom<&Args> for CompressorConfig {
    type Error = &'static str;

    fn try_from(args: &Args) -> Result<Self, Self::Error> {
        let default = CompressorConfig::default();
        CompressorConfig::new(
            default.start,
            default.end,
            default.special,
            default.disallowed,
            default.wordlist_delim,
            args.min_len,
            args.max_len,
            args.nproc,
            args.allow_overlaps,
            args.check_collisions,
        )
    }
}

fn extract_compressible_content(html: &str) -> &str {
    // Remove scripts (though in our case we expect clean HTML input)
    // For simplicity, we'll just extract content after the meta tag

    // Find the meta charset tag and extract everything after it
    let meta_pattern = Regex::new(r#"<meta charset="utf-8">(.*)"#).unwrap();

    if let Some(captures) = meta_pattern.captures(html) {
        captures.get(1).map_or(html, |m| m.as_str())
    } else {
        // If no meta tag found, try to find content after any doctype/html tags
        let re = "(?s)<!doctype[^>]*>.*?<meta charset=\"utf-8\">(.*)";
        let doctype_pattern = Regex::new(re).unwrap();
        if let Some(captures) = doctype_pattern.captures(html) {
            captures.get(1).map_or(html, |m| m.as_str())
        } else {
            html
        }
    }
}

fn get_io(args: &Args) -> io::Result<(Box<dyn Read>, Box<dyn Write>)> {
    // Validate that exactly one input and one output option is specified
    if !args.stdin && args.input_file.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Must specify either --stdin or --input-file",
        ));
    }

    if !args.stdout && args.output_file.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Must specify either --stdout or --output-file",
        ));
    }

    let input: Box<dyn Read> = if args.stdin {
        Box::new(io::stdin())
    } else if let Some(ref path) = args.input_file {
        Box::new(File::open(path)?)
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No input source specified",
        ));
    };

    let output: Box<dyn Write> = if args.stdout {
        Box::new(io::stdout())
    } else if let Some(ref path) = args.output_file {
        Box::new(File::create(path)?)
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No output destination specified",
        ));
    };

    Ok((input, output))
}

fn run(args: Args) -> io::Result<()> {
    let config = CompressorConfig::try_from(&args).unwrap();
    let (mut input_stream, mut output_stream) = get_io(&args)?;

    // Read input HTML
    let mut html = String::new();
    input_stream.read_to_string(&mut html)?;

    // Extract compressible content
    let text = extract_compressible_content(&html);

    if args.verbose {
        eprintln!("Extracted {} characters for compression", text.len());
    }

    // Note: The current compressor doesn't expose the config options yet
    // In a future iteration, we would pass the config to a configurable compressor
    // For now, we'll use the existing compress function
    let start = std::time::Instant::now();
    let compressed = config
        .compress(text.to_string())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    if args.verbose {
        eprintln!(
            "Compression completed in {} ms",
            start.elapsed().as_millis()
        );
    }

    // Write compressed output
    output_stream.write_all(compressed.as_bytes())?;

    if args.verbose {
        eprintln!("\nCompressed HTML written to output");
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    run(args)
}
