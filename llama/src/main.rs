use rand::{Rng, thread_rng};
use std::fs::File;
use std::io::{self, Read, Write};
use std::env;

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
  freq_cis_real: Vec<Vec<f32>>, // (seq_len, dim / 2)
  freq_cis_imag: Vec<Vec<f32>>, // (seq_len, dim / 2)
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
  let f32_buffer = read_f32_array(file, shape.0 * shape.1 * shape.2);
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
  let f32_buffer = read_f32_array(file, shape.0 * shape.1);
  let mut vec = vec![vec![0. ; shape.1]; shape.0];
  for i in 0..shape.0 {
    for j in 0..shape.1 {
      vec[i][j] = f32_buffer[i * shape.1 + j];
    }
  }
  return vec;
}

struct RunState {
  x: Vec<f32>, // activation of the current timestamp (dim, )
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

fn accum(x: &mut Vec<f32>, y: &Vec<f32>) {
  for i in 0..x.len() {
    x[i] += y[i];
  }
}

fn rmsnorm(out: &mut Vec<f32>, x: &Vec<f32>, weight: &Vec<f32>) {
  // root mean square normalization
  let mut ss = 0.;
  for i in 0..out.len(){
    ss += x[i] * x[i];
  }

  ss /=  x.len() as f32;
  ss += 1e-5;
  ss = 1. / ss.sqrt();
  for i in 0..out.len() {
    out[i] = weight[i] * (x[i] * ss);
  }
}


fn softmax(x: &mut [f32]) {
  let mut max_val = 0.;
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


fn matmul(out: &mut Vec<f32>, x: &Vec<f32>, w: &Vec<Vec<f32>>) {
  // W (d, n) @ (n, ) -> xout (d, )
  for i in 0..out.len() {
    let mut val = 0.;
    for j in 0..x.len() {
      val += w[i][j] * x[j];
    }
    out[i] = val;
  }
}

fn sample(probs: &Vec<f32>) -> usize {
  let mut rng = rand::thread_rng();
  let r: f32 = rng.gen_range(0.0..1.0);
  let mut cdf = 0.;
  for i in 0..probs.len() {
    cdf += probs[i];
    if r < cdf {
      return i;
    }
  }
  return probs.len() - 1;
}

fn transformer(token: usize, pos: usize, config: &Config, state: &mut RunState, weights: &mut TransformerWeights) {
  let dim = config.dim;
  let hidden_dim = config.hidden_dim;
  let head_size = dim / config.n_heads;

  let content_row = &weights.token_embedding_table[token];
  state.x.copy_from_slice(content_row);

  let freq_cis_real_row = &weights.freq_cis_real[pos];
  let freq_cis_imag_row = &weights.freq_cis_imag[pos];

  // forward all the layers
  for l in 0..config.n_layers {
    // attention rmsnorm
    rmsnorm(&mut state.xb, &state.x, &weights.rms_att_weight[l]);

    // compute attenttion scores
    // NOTE: the value of attention is let transformer aware of the context of a sentence
    // when it processes a token, they say it's like a look up table in the sense that
    // it look up the relevant informations of the "current" token with respect to its context
    matmul(&mut state.q, &state.xb, &weights.wq[l]);
    matmul(&mut state.k, &state.xb, &weights.wk[l]);
    matmul(&mut state.v, &state.xb, &weights.wv[l]);

    // NOTE: RoPE: rotary position embedding used to let transformer know the position of the word
    // it's processing with respect to the whole sentence
    // apply RoPE rotation to the q and k vectors for each head
    for h in 0..config.n_heads {
      // get the q and k vectors for this head
      let this_head_idx = h * head_size;

      // rotate q and k by the freq_cis_real and freq_cis_imag
      for i in (0..config.n_heads).step_by(2) {
        let q0 = state.q[this_head_idx + i];
        let q1 = state.q[this_head_idx + i+ 1];
        let k0 = state.k[this_head_idx + i];
        let k1 = state.k[this_head_idx + i+ 1];
        let fcr = freq_cis_real_row[i / 2 as usize];
        let fci = freq_cis_imag_row[i / 2 as usize];
        state.q[this_head_idx + i]   = q0 * fcr - q1 * fci;
        state.q[this_head_idx + i+1] = q0 * fci + q1 * fcr;
        state.k[this_head_idx + i]   = k0 * fcr - k1 * fci;
        state.k[this_head_idx + i+1] = k0 * fci + k1 * fcr;
      }

    }

    // save key, value as this time step (pos) to our kv cache
    state.key_cache[l][pos] = state.k.clone();
    state.value_cache[l][pos] = state.v.clone();

    // multihead attention, iterate over all heads
    for h in 0..config.n_heads {
      // get the query vector for this head
      let this_head_idx = h * head_size;
      // iterate over all timesteps, including the current one
      for t in 0..(pos + 1) {
        // get the key vector for this head and at this timestep
        let k = &state.key_cache[l][t][this_head_idx..this_head_idx + head_size];
        let mut score = 0.;
        for i in 0..head_size {
          score += state.q[this_head_idx + i] * k[i];
        }
        score /= (head_size as f32).sqrt();
        // save the score to the attention buffer
        state.att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(&mut state.att[0..pos + 1]);

      // weighted sum of the values, store back into xb
      for i in 0..head_size {
        let mut val = 0.;
        for t in 0..(pos + 1) {
          val += state.att[t] * state.value_cache[l][t][this_head_idx + i];
        }
        state.xb[this_head_idx + i] = val;
      }
    }

    // final matmul to get the output of the attention
    matmul(&mut state.xb2, &state.xb, &weights.wo[l]);

    // residual connection back into x
    accum(&mut state.x, &state.xb2);

    // ffn rmsnorm
    rmsnorm(&mut state.xb, &state.x, &weights.rms_ffn_weight[l]);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(&mut state.hb, &state.xb, &weights.w1[l]);
    matmul(&mut state.hb2, &state.xb, &weights.w3[l]);


    // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
    for i in 0..hidden_dim {
      state.hb[i] = state.hb[i] * (1. / (1. + (-state.hb[i]).exp()));
    }

    // elementwise multiply with w3(x)
    for i in 0..hidden_dim {
      state.hb[i] *= state.hb2[i];
    }

    // final matmul to get the output of the ffn
    matmul(&mut state.xb, &state.hb, &weights.w2[l]);

    // residual connection back into x
    accum(&mut state.x, &state.xb);
  }
  // final rmsnorm
  let x2 = state.x.clone();
  rmsnorm(&mut state.x, &x2, &weights.rms_final_weight);
  // classifier into logits
  matmul(&mut state.logits, &state.x, &weights.token_embedding_table);
}


const CONFIG_SIZE: usize = std::mem::size_of::<u32>() * 7;

fn main() {
  let args: Vec<String> = env::args().collect();
  let temperature = 0.9;
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
  let mut weights = TransformerWeights {
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

  let mut state = RunState {
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
    transformer(token, pos, &config, &mut state, &mut weights);
    if temperature == 0. {
      panic!("TODO: implement argmax");
    } else {
      for i in 0..vocab.len() {
        state.logits[i] /= temperature;
      }
      softmax(&mut state.logits);
      next = sample(&state.logits);
    }
    print!("{}", vocab[next]);
    io::stdout().flush().expect("Failed to flush stdout");
    token = next;
    pos += 1;
  }
  println!();

}
