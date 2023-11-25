struct Config {
  dim: u32,
  hidden_dim: u32,
  n_layers: u32,
  n_heads: u32,
  n_kv_heads: u32,
  vocab_size: u32,
  seq_len: u32,
}

struct TransformerWeights {
  // token embedding table
  // TODO this could be hardcoded and use an array instead of a vector
  token_embedding_table: Vec<Vec<f32>>, // (vocab_size, dim)
  // weights for rmsnorms
  rms_att_weight: Vec<Vec<f32>>, // (layers, dim)
  rms_ffn_weight: Vec<Vec<f32>>, // (layers, dim)
  // weights for matmuls
  wq: Vec<Vec<Vec<f32>>>, // (layers, dim, dim)
  wk: Vec<Vec<Vec<f32>>>, // (layers, dim, dim)
  wv: Vec<Vec<Vec<f32>>>, // (layers, dim, dim)
  wo: Vec<Vec<Vec<f32>>>, // (layers, dim, dim)
  // weights for ffn
  w1: Vec<Vec<Vec<f32>>>, // (layers, hidden_dim, dim)
  w2: Vec<Vec<Vec<f32>>>, // (layers, dim, hidden_dim)
  w3: Vec<Vec<Vec<f32>>>, // (layers, hidden_dim, dim)
  // final rmsnorm
  rms_final_weight: Vec<f32>, // (dim, )
  // freq_cis for RoPE relatively positional embeddings
  freq_cis_real: Vec<Vec<f32>>, // (seq_len, dim / 2)
  freq_cis_imag: Vec<Vec<f32>>, // (seq_len, dim / 2)
}

struct RunState {
  x: Vec<f32>, // activation of the current time stamp (dim, )
  xb: Vec<f32>, // same, but inside a residual branch (dim,)
  xb2: Vec<f32>, // an additional buffer just for convenience (dim,)
  hb: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
  hb2: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
  q: Vec<f32>, // query (dim,)
  k: Vec<f32>, // key (dim,)
  v: Vec<f32>, // value (dim,)
  att: Vec<f32>, // buffer for scores/attention values (seq_len,)
  logits: Vec<f32>, // output logits (vocab_size, )
  // kv cache
  key_cache: Vec<Vec<Vec<f32>>>,   // (layer, seq_len, dim)
  value_cache: Vec<Vec<Vec<f32>>>, // (layer, seq_len, dim)
}

fn load_weights() {

}

fn accumulate(x: &mut Vec<f32>, y: &Vec<f32>) {
  for i in 0..x.len() {
    x[i] += y[i];
  }
}

fn rmsnorm() {}


fn softmax(x: f32) {

}


fn matmul(x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
  // (a, b) @ (b, c)  => (a, c)
  let mut z = vec![vec![0.; y[0].len()]; x.len()];
  for i in 0..x.len() {
    for j in 0..y[0].len() {
      for k in 0..y.len() {
        z[i][j] += x[i][k] * y[k][j];
      }
    }
  }
  return z;
}

fn argmax(x: Vec<f32>) {

}

fn main() {
  let a = vec![vec![1., 2., 3.], vec![4., 5., 6.]];
  let b = vec![vec![7., 8., 1., 1.], vec![8., 10., 1., 1.], vec![11., 12., 1., 1.]];
  let c = matmul(&a, &b);
  println!("{:?}", c);
}
