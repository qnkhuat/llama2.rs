use std::io;
use std::io::prelude::*;
use std::fs::File;

fn main() {
  let mut f = File::open("playground.rs").unwrap();
  let mut buffer = [0; 10];
  let mut buffer1 = [0; 10];

  // read exactly 10 bytes
  f.read_exact(&mut buffer).unwrap();
  f.read_exact(&mut buffer1).unwrap();
  println!("{:?}, {:?}", buffer, buffer1);
}
