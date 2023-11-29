use std::fs::File;
use std::io::Read;
use std::env;
use std::mem;

#[derive(Debug)]
struct Config {
  dim: usize,
  hidden_dim: usize,
  n_layers: usize,
  n_heads: usize,
  n_kv_heads: usize,
  vocab_size: usize,
  seq_len: usize,
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
  freq_cis_real: Vec<Vec<f32>>, // (seq_len, dim / n_heads / 2)
  freq_cis_imag: Vec<Vec<f32>>, // (seq_len, dim / n_heads / 2)
}

fn read_f32_array(file: &mut File, size: usize) -> Vec<f32>{
  let mut buffer = vec![0; size * 4];
  file.read_exact(&mut buffer).unwrap();
  return buffer.chunks(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect::<Vec<f32>>();
}

fn read_string(file: &mut File, size: usize) -> String {
  let mut buffer = vec![0; size];
  file.read_exact(&mut buffer).unwrap();
  return String::from_utf8(buffer).unwrap();
}

fn read_3d_vec(file: &mut File, shape: (usize, usize, usize)) -> Vec<Vec<Vec<f32>>> {
  let mut f32_buffer = read_f32_array(file, shape.0 * shape.1 * shape.2);
  let mut vec = vec![vec![vec![0.; shape.2]; shape.1]; shape.0];
  for i in 0..shape.0 {
    for j in 0..shape.1 {
      for k in 0..shape.2 {
        vec[i][j][k] = f32_buffer[i * shape.1 * shape.2 + j * shape.2 + k];
      }
    }
  }
  return vec;
}

fn read_2d_vec(file: &mut File, shape: (usize, usize)) -> Vec<Vec<f32>> {
  let mut f32_buffer = read_f32_array(file, shape.0 * shape.1);
  let mut vec = vec![vec![0. ; shape.1]; shape.0];
  for i in 0..shape.0 {
    for j in 0..shape.1 {
      vec[i][j] = f32_buffer[i * shape.1 + j];
    }
  }
  return vec;
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

//fn load_weights(file: &mut File) -> TransformerWeights{
//
//}

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
const CONFIG_SIZE: usize = std::mem::size_of::<u32>() * 7;

fn main() {
  let args: Vec<String> = env::args().collect();
  assert!(args.len() == 2, "you must provide a model file");
  let model_file = &args[1];
  // assert if file exists
  assert!(std::path::Path::new(model_file).exists(), "model file not found");
  let mut file = File::open(model_file).unwrap();
  let mut config_buffer = [0; CONFIG_SIZE];
  file.read_exact(&mut config_buffer).unwrap();
  let raw_config = unsafe { std::mem::transmute::<[u8; CONFIG_SIZE], [i32; 7]>(config_buffer) };
  let config = Config {
    dim: raw_config[0] as usize,
    hidden_dim: raw_config[1] as usize,
    n_layers: raw_config[2] as usize,
    n_heads: raw_config[3] as usize,
    n_kv_heads: raw_config[4] as usize,
    vocab_size: raw_config[5] as usize,
    seq_len: raw_config[6] as usize,
  };
  println!("Config: {:?}", config);
  // load vocab
  let mut vocab: Vec<String>= vec![String::new(); config.vocab_size];
  let mut vocab_file = File::open("tokenizer.bin").unwrap();
  for i in 0..config.vocab_size {
    let mut len_buf = [0; 4];
    vocab_file.read_exact(&mut len_buf).unwrap();
    let len = i32::from_le_bytes(len_buf);
    vocab[i] = read_string(&mut vocab_file, len as usize);
  }

  // load weights;
  let head_size = config.dim / config.n_heads;
  let w = TransformerWeights {
    token_embedding_table: read_2d_vec(&mut file, (config.vocab_size, config.dim)),
    rms_att_weight: read_2d_vec(&mut file, (config.n_layers, config.dim)),
    wq: read_3d_vec(&mut file, (config.n_layers, config.dim, config.dim)),
    wk: read_3d_vec(&mut file, (config.n_layers, config.dim, config.dim)),
    wv: read_3d_vec(&mut file, (config.n_layers, config.dim, config.dim)),
    wo: read_3d_vec(&mut file, (config.n_layers, config.dim, config.dim)),
    rms_ffn_weight: read_2d_vec(&mut file, (config.n_layers, config.dim)),
    w1: read_3d_vec(&mut file, (config.n_layers, config.hidden_dim, config.dim)),
    w2: read_3d_vec(&mut file, (config.n_layers, config.dim, config.hidden_dim)),
    w3: read_3d_vec(&mut file, (config.n_layers, config.hidden_dim, config.dim)),
    rms_final_weight: read_f32_array(&mut file, config.dim),
    freq_cis_real: read_2d_vec(&mut file, (config.seq_len, (head_size / 2) as usize)),
    freq_cis_imag: read_2d_vec(&mut file, (config.seq_len, (head_size / 2) as usize)),
  };
}
