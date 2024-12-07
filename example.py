from types import MethodType
import utils

##### Prep ######
# get model and tokenizer
model, tokenizer = utils.load_model() # default is Qwen 1.5B because LlaMa too heavy, can change by doing model_path="model/from/huggingface"

#### Embedding ####

# embed text using model embedding
print("\nCalculated embedding. Feel free to modify after this.")
ex_embedding = utils.embed_text(model, tokenizer, "Add text here")


##### Ways to run the model ######

# >>>> 1. Show topk probabilities of output token for each position <<<<``
print("\n\nTop_k (k=5) probabilities for position 0 of output")

# print top_k=5 for first position
tok_position = 0
utils.topk_probabilities(model, tokenizer, ex_embedding, tok_position, k=5)
print(f"\t NOTE: use \"utils.topk_probabilities\" for this")


# >>>> 2. Regular text output (default for LLMs) <<<<
print("\n\n")

# Get regular text ouptut of model
text_output = utils.get_text_output(model, tokenizer, ex_embedding)
print(f"Text output: \"{text_output[0].strip()}\" \n\t NOTE: use \"utils.get_text_output\" for this")
