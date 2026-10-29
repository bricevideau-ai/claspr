# Karpathy's llama2.c `stories15M` checkpoint, as a claspr model spec.
#
# The config lines must match the checkpoint header. The graph section is read
# top to bottom and built into ONE claspr graph that computes one token: each op
# maps the residual stream to itself, `repeat` stamps its body once per layer,
# and weight names refer to checkpoint tensors (per-layer ones are indexed by
# the current `repeat` iteration).

dim 288
hidden_dim 768
n_layers 6
n_heads 6
n_kv_heads 6
vocab_size 32000
seq_len 256

embed token_embedding
repeat n_layers
  attention rms_att wq wk wv wo
  ffn rms_ffn w1 w2 w3
end
rmsnorm rms_final
classifier token_embedding
argmax
