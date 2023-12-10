# Inference llama2 in one file of pure Rust
Taking the inspiration from https://github.com/karpathy/llama2.c

# To run
Download the weights
```
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Build the binary
```
rustc run.rs -C opt-level=3 -o run.out
```

Run inference
```
./run.out stories15M.bin
```

# Performance
Currently the performance is shit, it's generating ~5tok/s on my M1 Max


| Commit | Tok / s | Remarks |
|--------|---------|-------------|
| [3774d76](https://github.com/qnkhuat/llama2.rs/commit/3774d76) | 111 | Use 1-D vector instead of multi dimension vector |
| [1a3b83e](https://github.com/qnkhuat/llama2.rs/commit/1a3b83e) | 75 | build with opt-level=3 |
| [ab08f7e](https://github.com/qnkhuat/llama2.rs/blob/ab08f7e/run.c) | 5.2 | The very first version |
