import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from types import MethodType

def load_model(model_path="Qwen/Qwen2.5-1.5B") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads model from HuggingFace based on model_path.
    Returns:
        tuple: A tuple containing the model with disabled embedding layer loaded on cuda or cpu accordingly, and the tokenizer.
    """

    # disable the embedding layer
    def custom_forward(self, inputs_embeds=None, **kwargs):
        # pass embeddings directly to the transformer layers
        return self.model(inputs_embeds=inputs_embeds, **kwargs)

    loaded_on = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model \"{model_path}\" on {loaded_on}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=not(torch.cuda.is_available()),  #  use low mem if no gpu :(
    )

    model.forward = MethodType(custom_forward, model)
    model = model.to(0 if torch.cuda.is_available() else "cpu")
    print(f"Model loaded.")


    return model, tokenizer
def embed_text(model, tokenizer, text: str):
    """
    Tokenizes the input text and computes the embeddings using the provided model and tokenizer.
    
    Args:
        model: The pre-trained model to use for embedding.
        tokenizer: The tokenizer to use for tokenizing the input text.
        text (str): The input text to be tokenized and embedded.
    
    Returns:
        torch.Tensor: The computed embeddings for the input text.
    """
def embed_text(model, tokenizer, text: str):
    # tokenize + embed
    inputs = tokenizer(text, return_tensors='pt')
    precomputed_embeddings = model.model.embed_tokens(inputs["input_ids"])

    # move embeddings to cpu/gpu
    precomputed_embeddings = precomputed_embeddings.to(0 if torch.cuda.is_available() else "cpu")
    loaded_on = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embeddings loaded on {loaded_on}")

    return precomputed_embeddings


def _get_logits(model, embedding):
    """
    Computes the logits given the model and the input embeddings.
    
    Args:
        model: The pre-trained model to use for computing logits.
        embedding: The input embeddings for which to compute the logits.
    
    Returns:
        torch.Tensor: The computed logits for the input embeddings.
    """
    outputs = model(inputs_embeds=embedding)
    return model.lm_head(outputs.last_hidden_state)

def topk_probabilities(model, tokenizer, embedding, tok_position, k=5):
    """
    Computes and prints the top-k highest probability tokens at a specified position in the input embedding.
    Args:
        model (torch.nn.Module): The language model used to generate logits.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to decode token IDs.
        embedding (torch.Tensor): The input embedding for which probabilities are computed.
        tok_position (int): The position in the embedding to consider for top-k probabilities.
        k (int, optional): The number of top probabilities to retrieve. Defaults to 5.
    Returns:
        None
    Prints:
        The top-k tokens and their corresponding probabilities in descending order.
    """
    
    # get probs
    logits = _get_logits(model, embedding)
    probabilities = torch.softmax(logits, dim=-1)

    # pull highest probability tokens for x positions
    vals = torch.topk(probabilities[0][tok_position], k=k).values
    ids = torch.topk(probabilities[0][tok_position], k=k)[1:]
    for _, i in enumerate(list(ids[0].tolist())):
        print(f"#{_+1} highest: \"{tokenizer.decode(i)}\" ({vals[_]*100:.2f}%)")

def get_text_output(model, tokenizer, embedding):
    """
    Converts model outputs into human-readable text using the provided tokenizer.

    Args:
        model: A pre-trained language model that processes the embeddings
        tokenizer: A tokenizer object used for decoding token IDs into text
        embedding: The input embedding tensor to be processed by the model

    Returns:
        list[str]: A list of decoded text strings, one for each sequence in the batch

    Note:
        - The function expects the model to return logits that can be converted to token IDs
        - The output text has special tokens removed due to skip_special_tokens=True
        - The function processes the entire batch at once and returns decoded text for all sequences
    """

    logits = _get_logits(model, embedding)

    predicted_token_ids = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)
    decoded_text = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

    return decoded_text