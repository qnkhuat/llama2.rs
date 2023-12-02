use std::io;
use std::io::prelude::*;
use std::fs::File;

fn softmax(x: &mut [f32]) {
  let mut max_val = x[0];
  for i in 1..x.len() {
    if x[i] > max_val {
      max_val = x[i];
    }
  }
  // exp and sum
  let mut sum = 0.;
  for i in 0..x.len() {
    x[i] = (x[i] - max_val).exp();
    sum += x[i];
  }
  // normalize
  for i in 0..x.len() {
    x[i] /= sum;
  }
}

fn main() {

  let mut x = vec![1., 2., 3.];
  softmax(&mut x[0..2]);
  println!("{:?}", x);
}
