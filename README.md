# research_kimon_raman

#### Recommendations:
- at least 16gb RAM
- powerful cpu


#### `outputs`: contains multiple tensors. 
- outputs.last_hidden_state: represents the model's internal representation of the text at each token position
- note: hidden states are then transformed into word probabilities (logits) using the language model head (lm_head)

#### `logits`: a tensor representing the probability distributions over the vocabulary for each token in the sequence

Logits Shape: (batch_size, seq_len, vocab_size)
- batch_size: Number of sequences processed in parallel.
- seq_len: Number of tokens in the input sequence.
- vocab_size: Number of tokens in the model's vocabulary.

#### `probabilities`: the distributed probabilities across vocab size computed using softmax of logits