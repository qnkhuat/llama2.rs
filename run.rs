// rustc run.rs -C opt-level=3 -o run.out  && ./run.out model.bin
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

//{ dim: 288, hidden_dim: 768, n_layers: 6, n_heads: 6, n_kv_heads: 6, vocab_size: 32000, seq_len: 256 }
const DIM : usize = 288;
const HIDDEN_DIM : usize = 768;
const N_LAYERS : usize = 6;
const N_HEADS : usize = 6;
const N_KV_HEADS : usize = 6;
const VOCAB_SIZE : usize = 32000;
const SEQ_LEN : usize = 256;

struct TransformerWeights {
  // token embedding table
  // TODO this could be hardcoded and use an array instead of a vector
  token_embedding_table: [f32; DIM * VOCAB_SIZE], // (vocab_size, dim)
  // weights for rmsnorms
  rms_att_weight: [f32; N_LAYERS * DIM], // (layers, dim)
  rms_ffn_weight: [f32; N_LAYERS * DIM], // (layers, dim)
  // weights for matmuls
  wq: [f32; N_LAYERS * DIM * DIM], // (layers, dim, dim)
  wk: [f32; N_LAYERS * DIM * DIM], // (layers, dim, dim)
  wv: [f32; N_LAYERS * DIM * DIM], // (layers, dim, dim)
  wo: [f32; N_LAYERS * DIM * DIM], // (layers, dim, dim)
  // weights for ffn
  w1: [f32; N_LAYERS * HIDDEN_DIM * DIM], // (layers, hidden_dim, dim)
  w2: [f32; N_LAYERS * DIM * HIDDEN_DIM], // (layers, dim, hidden_dim)
  w3: [f32; N_LAYERS * HIDDEN_DIM * DIM], // (layers, hidden_dim, dim)
  // final rmsnorm
  rms_final_weight: [f32; DIM], // (dim, )
  // freq_cis for RoPE relatively positional embeddings
  freq_cis_real: [f32; SEQ_LEN * DIM / N_HEADS / 2], // (seq_len, dim / n_heads / 2)
  freq_cis_imag: [f32; SEQ_LEN * DIM / N_HEADS / 2], // (seq_len, dim / n_heads / 2)
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

struct RunState {
  x: [f32; DIM], // activation of the current timestamp (dim, )
  xb: [f32; DIM], // same, but inside a residual branch (dim,)
  xb2: [f32; DIM], // an additional buffer just for convenience (dim,)
  hb: [f32; HIDDEN_DIM], // buffer for hidden dimension in the ffn (hidden_dim,)
  hb2: [f32; HIDDEN_DIM], // buffer for hidden dimension in the ffn (hidden_dim,)
  q: [f32; DIM], // query (dim,)
  k: [f32; DIM], // key (dim,)
  v: [f32; DIM], // value (dim,)
  att: [f32; SEQ_LEN], // buffer for scores/attention values (seq_len,)
  logits: [f32; VOCAB_SIZE], // output logits (vocab_size, )
  // kv cache
  key_cache: [f32; N_LAYERS * SEQ_LEN * DIM], // (layer, seq_len, dim)
  value_cache: [f32; N_LAYERS * SEQ_LEN * DIM], // (layer, seq_len, dim)
}

fn accum(x: &mut [f32], y: &[f32]) {
  for i in 0..x.len() {
    x[i] += y[i];
  }
}

fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
  // root mean square normalization
  let mut ss = 0.;
  for item in x.iter() {
    ss += item * item;
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

fn matmul(out: &mut [f32], w: &[f32], x: &[f32]) {
  // W (d, n) @ (n, ) -> xout (d, )
  let n = x.len();
  let d = out.len();
  for i in 0..d {
    let mut val = 0.;
    for j in 0..n {
      val += w[i * n + j] * x[j];
    }
    out[i] = val;
  }
}
//use rand::{Rng, thread_rng};
//fn sample(probs: &Vec<f32>) -> usize {
//  let mut rng = rand::thread_rng();
//  let r: f32 = rng.gen_range(0.0..1.0);
//  let mut cdf = 0.;
//  for i in 0..probs.len() {
//    cdf += probs[i];
//    if r < cdf {
//      return i;
//    }
//  }
//  return probs.len() - 1;
//}

fn argmax(probs: &[f32]) -> usize {
  let mut max_i = 0;
  let mut max_p = probs[0];
  for i in 0..probs.len() {
    if probs[i] > max_p {
      max_i = i;
      max_p = probs[i];
    }
  }
  return max_i
}

fn transformer(token: usize, pos: usize, config: &Config, state: &mut RunState, weights: &mut TransformerWeights) {
  let dim = config.dim;
  let hidden_dim = config.hidden_dim;
  let head_size = dim / config.n_heads;

  let content_row = &weights.token_embedding_table[token * dim .. (token + 1) * dim];
  state.x.copy_from_slice(content_row);

  let freq_cis_real_row = &weights.freq_cis_real[pos * head_size / 2.. (pos + 1) * head_size / 2];
  let freq_cis_imag_row = &weights.freq_cis_imag[pos * head_size / 2.. (pos + 1) * head_size / 2];

  // forward all the layers
  for l in 0..config.n_layers {
    // attention rmsnorm
    rmsnorm(&mut state.xb, &state.x, &weights.rms_att_weight[l * dim .. (l + 1) * dim]);

    // compute attenttion scores
    // NOTE: the value of attention is let transformer aware of the context of a sentence
    // when it processes a token, they say it's like a look up table in the sense that
    // it look up the relevant informations of the "current" token with respect to its context
    matmul(&mut state.q, &weights.wq[l * dim * dim .. (l + 1) * dim * dim], &state.xb);
    matmul(&mut state.k, &weights.wk[l * dim * dim .. (l + 1) * dim * dim], &state.xb);
    matmul(&mut state.v, &weights.wv[l * dim * dim .. (l + 1) * dim * dim], &state.xb);

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
        let fcr = freq_cis_real_row[i / 2];
        let fci = freq_cis_imag_row[i / 2];
        state.q[this_head_idx + i]   = q0 * fcr - q1 * fci;
        state.q[this_head_idx + i+1] = q0 * fci + q1 * fcr;
        state.k[this_head_idx + i]   = k0 * fcr - k1 * fci;
        state.k[this_head_idx + i+1] = k0 * fci + k1 * fcr;
      }
    }

    // save key, value as this time step (pos) to our kv cache
    let loff = l * config.seq_len * dim;
    state.key_cache[loff + pos * dim .. loff + (pos + 1) * dim].copy_from_slice(&state.k);
    state.value_cache[loff + pos * dim .. loff + (pos + 1) * dim].copy_from_slice(&state.v);

    // multihead attention, iterate over all heads
    for h in 0..config.n_heads {
      // get the query vector for this head
      let this_head_idx = h * head_size;
      // iterate over all timesteps, including the current one
      for t in 0..(pos + 1) {
        // get the key vector for this head and at this timestep
        let k = &state.key_cache[loff + t * dim + this_head_idx .. loff + t * dim + this_head_idx + head_size];

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
          val += state.att[t] * state.value_cache[loff + t * dim + this_head_idx + i];
        }
        state.xb[this_head_idx + i] = val;
      }
    }

    // final matmul to get the output of the attention
    matmul(&mut state.xb2, &weights.wo[l * dim * dim .. (l + 1) * dim * dim], &state.xb);

    // residual connection back into x
    accum(&mut state.x, &state.xb2);

    // ffn rmsnorm
    rmsnorm(&mut state.xb, &state.x, &weights.rms_ffn_weight[l * dim .. (l + 1) * dim]);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(&mut state.hb, &weights.w1[l * hidden_dim * dim .. (l + 1) * hidden_dim * dim], &state.xb);
    matmul(&mut state.hb2, &weights.w3[l * hidden_dim * dim .. (l + 1) * hidden_dim * dim], &state.xb);

    // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
    for i in 0..hidden_dim {
      state.hb[i] = state.hb[i] * (1. / (1. + (-state.hb[i]).exp()));
    }

    // elementwise multiply with w3(x)
    for i in 0..hidden_dim {
      state.hb[i] *= state.hb2[i];
    }

    // final matmul to get the output of the ffn
    matmul(&mut state.xb, &weights.w2[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim], &state.hb);

    // residual connection back into x
    accum(&mut state.x, &state.xb);
  }
  // final rmsnorm
  let x2 = state.x.clone();
  rmsnorm(&mut state.x, &x2, &weights.rms_final_weight);
  // classifier into logits
  matmul(&mut state.logits, &weights.token_embedding_table, &state.x);
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
  let mut weights = TransformerWeights {
    token_embedding_table: {
      let data = read_f32_array(&mut file, config.vocab_size * config.dim);
      let mut table = [0.; DIM * VOCAB_SIZE];
      table.copy_from_slice(&data);
      table
    },
    rms_att_weight: {
      let data = read_f32_array(&mut file, config.n_layers * config.dim);
      let mut table = [0.; N_LAYERS * DIM];
      table.copy_from_slice(&data);
      table
    },
    wq: {
      let data = read_f32_array(&mut file, config.n_layers * config.dim * config.dim);
      let mut table = [0.; N_LAYERS * DIM * DIM];
      table.copy_from_slice(&data);
      table
    },
    wk: {
      let data = read_f32_array(&mut file, config.n_layers * config.dim * config.dim);
      let mut table = [0.; N_LAYERS * DIM * DIM];
      table.copy_from_slice(&data);
      table
    },
    wv: {
      let data = read_f32_array(&mut file, config.n_layers * config.dim * config.dim);
      let mut table = [0.; N_LAYERS * DIM * DIM];
      table.copy_from_slice(&data);
      table
    },
    wo: {
      let data = read_f32_array(&mut file, config.n_layers * config.dim * config.dim);
      let mut table = [0.; N_LAYERS * DIM * DIM];
      table.copy_from_slice(&data);
      table
    },
    rms_ffn_weight: {
      let data = read_f32_array(&mut file, config.n_layers * config.dim);
      let mut table = [0.; N_LAYERS * DIM];
      table.copy_from_slice(&data);
      table
    },
    w1: {
      let data = read_f32_array(&mut file, config.n_layers * config.dim * config.hidden_dim);
      let mut table = [0.; N_LAYERS * HIDDEN_DIM * DIM];
      table.copy_from_slice(&data);
      table
    },
    w2: {
      let data = read_f32_array(&mut file, config.n_layers * config.hidden_dim * config.dim);
      let mut table = [0.; N_LAYERS * DIM * HIDDEN_DIM];
      table.copy_from_slice(&data);
      table
    },
    w3: {
      let data = read_f32_array(&mut file, config.n_layers * config.dim * config.hidden_dim);
      let mut table = [0.; N_LAYERS * HIDDEN_DIM * DIM];
      table.copy_from_slice(&data);
      table
    },
    rms_final_weight: {
      let data = read_f32_array(&mut file, config.dim);
      let mut table = [0.; DIM];
      table.copy_from_slice(&data);
      table
    },
    freq_cis_real: {
      let data = read_f32_array(&mut file, config.seq_len * (head_size / 2) as usize);
      let mut table = [0.; SEQ_LEN * DIM / N_HEADS / 2];
      table.copy_from_slice(&data);
      table
    },
    freq_cis_imag: {
      let data = read_f32_array(&mut file, config.seq_len * (head_size / 2) as usize);
      let mut table = [0.; SEQ_LEN * DIM / N_HEADS / 2];
      table.copy_from_slice(&data);
      table
    },
  };

  //println!("weights loaded");

  //drop(file);

  //// load vocab
  //let mut vocab: Vec<String>= vec![String::new(); config.vocab_size];
  //{
  //  let mut vocab_file = File::open("tokenizer.bin").unwrap();
  //  for i in 0..config.vocab_size {
  //    let mut len_buf = [0; 4];
  //    vocab_file.read_exact(&mut len_buf).unwrap();
  //    let len = i32::from_le_bytes(len_buf);
  //    vocab[i] = read_string(&mut vocab_file, len as usize);
  //  }
  //}

  //let mut state = RunState {
  //  x: [0.; DIM],
  //  xb: [0.; DIM],
  //  xb2: [0.; DIM],
  //  hb: [0.; HIDDEN_DIM],
  //  hb2: [0.; HIDDEN_DIM],
  //  q: [0.; DIM],
  //  k: [0.; DIM],
  //  v: [0.; DIM],
  //  att: [0.; SEQ_LEN],
  //  logits: [0.; VOCAB_SIZE],
  //  key_cache: [0.; DIM * SEQ_LEN * N_LAYERS],
  //  value_cache: [0.; DIM * SEQ_LEN * N_LAYERS],
  //};

  //let mut next: usize;
  //let mut token = 1; // 1 = BOS token in Llama-2 sentencepiece
  //let mut pos = 0;
  //let start = std::time::Instant::now();
  //while pos < config.seq_len {
  //  transformer(token, pos, &config, &mut state, &mut weights);
  //  next = argmax(&state.logits);
  //  print!("{}", vocab[next]);
  //  io::stdout().flush().expect("Failed to flush stdout");
  //  token = next;
  //  pos += 1;
  //}
  //println!();

  //println!("tok/s = {}", config.seq_len as f32 / start.elapsed().as_secs_f32());
}
