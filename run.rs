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
  freq_cis_real: Vec<Vec<f32>>, // (dim / n_heads / 2, seq_len)
  freq_cis_imag: Vec<Vec<f32>>, // (dim / n_heads / 2, seq_len)
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

fn accumulate(x: &mut Vec<f32>, y: &Vec<f32>) {
  for i in 0..x.len() {
    x[i] += y[i];
  }
}

fn rmsnorm(out: &mut Vec<f32>, x: &Vec<f32>, weight: &Vec<f32>) {
  // root mean square normalization
  let ss = 0.;
  for i in 0..out.len(){
    ss += x[i] * x[i];
  }

  ss /= o.len() as f32;
  ss += 1e-5;
  ss = 1. / std::num::sqrt(ss);
  for i in 0..out.len() {
    out[i] = x[i] * ss * weight[i];
  }
}


fn softmax(x: f32) {

}


fn matmul(out: &mut Vec<f32>, x: &Vec<f32>, w: &Vec<Vec<f32>>) {
  // W (d, n) @ (n, ) -> xout (d, )
  for i in 0..out.len() {
    out[i] = 0.;
    for j in 0..x.len() {
      out[i] += w[i][j] * x[j];
    }
  }
}

fn argmax(x: Vec<f32>) {

}


fn transformer(token: usize, pos: usize, config: &Config, state: &mut RunState, weights: &mut TransformerWeights) -> usize {
  let dim = config.dim;
  let hidden_dim = config.hidden_dim;
  let head_size = dim / config.n_heads;

  let content_row = weights.token_embedding_table[token];
  let x = content_row.clone();

  let freq_cis_real_row = &w.freq_cis_real[pos];
  let freq_cis_imag_row = &w.freq_cis_imag[pos];

  // forward all the layers
  for l in 0..config.n_layers {
    // attention rmsnorm
    rmsnorm(&mut state.xb, &state.x, &weights.rms_att_weight[l]);

    // qkv matmuls for this position
    matmul(&mut state.q, &state.xb, &weights.wq[l]);
    matmul(&mut state.k, &state.xb, &weights.wk[l]);
    matmul(&mut state.v, &state.xb, &weights.wv[l]);

    // apply RoPE rotation to the q and k vectors for each head
    for h in 0..config.n_heads {
      // get the q and k vectors for this head
      let q = &mut state.q[h];
      let k = &mut state.k[h];

      // rotate q and k by the freq_cis_real and freq_cis_imag
      for i in (0..config.n_heads).step_by(2) {
        let q0 = q[i];
        let q1 = q[i+ 1];
        let k0 = k[i];
        let k1 = k[i+ 1];
        let fcr = freq_cis_real_row[i / 2 as usize];
        let fci = freq_cis_imag_row[i / 2 as usize];
        q[i]   = q0 * fcr - q1 * fci;
        q[i+1] = q0 * fci + q1 * fcr;
        k[i]   = k0 * fcr - k1 * fci;
        k[i+1] = k0 * fci + k1 * fcr;
      }
    }

    // save key, value as this time step (pos) to our kv cache
    // TODO

  }

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
    freq_cis_real: read_2d_vec(&mut file, ((head_size / 2) as usize, config.seq_len)),
    freq_cis_imag: read_2d_vec(&mut file, ((head_size / 2) as usize, config.seq_len)),
  };

  drop(file);

  // load vocab
  let mut vocab: Vec<String>= vec![String::new(); config.vocab_size];
  {
    let mut vocab_file = File::open("tokenizer.bin").unwrap();
    for i in 0..config.vocab_size {
      let mut len_buf = [0; 4];
      vocab_file.read_exact(&mut len_buf).unwrap();
      let len = i32::from_le_bytes(len_buf);
      vocab[i] = read_string(&mut vocab_file, len as usize);
    }
  }

  let state = RunState {
    x: vec![0.; config.dim],
    xb: vec![0.; config.dim],
    xb2: vec![0.; config.dim],
    hb: vec![0.; config.hidden_dim],
    hb2: vec![0.; config.hidden_dim],
    q: vec![0.; config.dim],
    k: vec![0.; config.dim],
    v: vec![0.; config.dim],
    att: vec![0.; config.seq_len],
    logits: vec![0.; config.vocab_size],
    key_cache: vec![vec![vec![0.; config.dim]; config.seq_len]; config.n_layers],
    value_cache: vec![vec![vec![0.; config.dim]; config.seq_len]; config.n_layers],
  };


  let mut next: usize;
  let mut token = 1; // 1 = BOS token in Llama-2 sentencepiece
  let mut pos = 0;
  while pos < config.seq_len {
    //transformer(token, pos, config, state, weights);

  }


}
