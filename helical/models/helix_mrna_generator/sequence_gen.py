import numpy as np
from tqdm import trange
from helical.models.helix_mrna_generator.model.modelling_helix_mrna import HelixmRNAForCausalLM
from helical.models.helix_mrna_generator.model.hg38_char_tokenizer import CharTokenizer
import torch
import random
import torch.nn.functional as F

# TODO: unite with regular helix?

def softmax_with_temperature(logits, temperature):
    """
    Apply softmax with temperature scaling to the logits

    Parameters
    ----------
    logits : torch.Tensor
        Logits. Shape (batch_size, vocab_size)
    temperature : float
        Temperature. Lower values will make the distribution more deterministic which encourages exploitation
        Higher values will make the distribution more uniform which encourages exploration
    """
    if temperature <= 0:
        return torch.nn.functional.softmax(logits, dim=-1)
    return torch.nn.functional.softmax(logits / temperature, dim=-1)

def sample_next_token(
    token_logits,
    softmax_temperature,
    definite_selection_threshold,
    top_k,
    top_p,
):
    """
    Sample the next token from the token logits

    Notes
    ----------
    This uses a combination of top-k sampling and temperature scaling to sample the next token from the token logits

    Parameters
    ----------
    token_logits : torch.Tensor
        Token logits. Sheape (batch_size, vocab_size)
    softmax_temperature : float
        Softmax temperature
    definite_selection_threshold : float
        Definite selection threshold
    top_k : int
        Number of top selected probabilities to consider

    Returns
    ----------
    int
        Next token selected
    float
        Logit value of the next token selected
    """
    token_logits_with_temp = softmax_with_temperature(
        token_logits, softmax_temperature
    ) # (batch_size, vocab_size)
    batch_size = token_logits_with_temp.shape[0]

    topk = token_logits_with_temp.topk(top_k, dim=-1)
    top_values = topk.values  # (batch_size, K)
    top_indices = topk.indices  # (batch_size, K)

    sampled_indices = torch.zeros(batch_size, dtype=torch.long, device=token_logits.device)
    
    for i in range(batch_size):

        if (
            token_logits_with_temp[i][top_indices[i, 0]].item()
            < definite_selection_threshold
        ):
            top_logits = top_values[i]
            if top_p > 0.0:
                top_logits = torch.cumsum(top_logits, dim=-1)
                sorted_indices = torch.arange(
                    start=top_logits.shape[-1] - 1,
                    end=-1,
                    step=-1,
                    device=top_logits.device,
                )
                sorted_indices = sorted_indices[top_logits[sorted_indices] < top_p]
                top_logits = top_logits[sorted_indices]
            if top_logits.shape[-1] <= 1:
                sampled_indices[i] = 0
            else:
                sampled_indices[i] = torch.multinomial(top_logits, 1).item()
        else:
            sampled_indices[i] = 0 # greedy

    rows = torch.arange(batch_size, device=token_logits.device)
    chosen_tokens = top_indices[rows, sampled_indices]
    chosen_scores = token_logits_with_temp[rows, chosen_tokens]

    return chosen_tokens, chosen_scores
    

def append_next_token(input_data, new_value):
    new_tensor = new_value.clone().detach().unsqueeze(0).to("cuda")

    input_data = {
        "input_ids": torch.cat([input_data["input_ids"][0], new_tensor], dim=-1)
    }

    input_data["input_ids"] = input_data["input_ids"].unsqueeze(0)

    return input_data

def generate(
    helix_model_lm,
    tokenizer,
    max_length_to_generate: int = 100,
    softmax_temperature: float = 0.6,
    definite_selection_threshold: float = 0.8,
    top_k: int = 3,
    top_p: float = 0.0,
):
    generated_sequence = {"sequence": [], "scores": []}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            start_letter = random.choice(["A", "C", "G", "U"])
            input_data = tokenizer(
                start_letter,
                return_tensors="pt",
            )
            max_length_to_generate -= 1

            input_data["input_ids"] = input_data["input_ids"][:, :-1]

            input_data.to("cuda")

            scores = torch.zeros_like(input_data["input_ids"]).squeeze(0)

            for _ in trange(max_length_to_generate):
                logits = helix_model_lm(**input_data)[0][0][-1]
                logits = logits.unsqueeze(0)  # shape (1, vocab_size)

                new_value, score = sample_next_token(
                    logits,
                    softmax_temperature,
                    definite_selection_threshold,
                    top_k,
                    top_p,
                )
                new_value = new_value.squeeze(0)
                score = score.squeeze(0)

                input_data = append_next_token(input_data, new_value)
                scores = torch.cat([scores, score.unsqueeze(0)])

            sequence = tokenizer.decode(input_data["input_ids"].squeeze())

            generated_sequence["sequence"].append(sequence)
            generated_sequence["scores"].append(scores.cpu().float().numpy())

    return generated_sequence


def compute_sequence_log_likelihood(model, sequences, tokenizer, device="cuda"):
    """
    Compute total log-likelihood for a batch of sequences.
    Assumes input_ids is of shape (batch_size, seq_len) and contains token IDs.
    """
    input_ids = tokenizer(sequences, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids).logits  # (batch_size, seq_len, vocab_size) 
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # convert logits to log probabilities, excluding the last token (batch_size, seq_len - 1, vocab_size)
        targets = input_ids[:, 1:]  # shift labels by one
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len - 1)
        sequence_log_likelihood = token_log_probs.sum(dim=1)  # (batch_size,)
    return sequence_log_likelihood

def mutate_sequence(
    model,
    device,
    tokenizer,
    mutation_position: int,
    original_seq: list[str],
    mutation_length: int = 5,
    softmax_temperature: float = 0.6,
    top_k: int = 3,
    logits_threshold: float = 0.8,
    top_p: float = 0.0, # 0.0 = no top-p samplings
):
    input_data = tokenizer(original_seq, return_tensors="pt")
    input_data["input_ids"] = input_data["input_ids"][:, :-1] # exclude SEP if present
    input_data.to(device)
    seq_len = input_data["input_ids"].shape[1]

    # Choose a mutation window
    if mutation_position not in range(seq_len - mutation_length + 1):
        raise ValueError(
            f"Mutation position {mutation_position} is out of range for sequence of length {seq_len}"
        )
    else: 
        start = mutation_position
    prefix = input_data["input_ids"][:, :start].to(device)
    batch_size = prefix.shape[0]
    scores = torch.zeros((batch_size, mutation_length)).to(device)
    
    # Ensure prefix is non-empty
    start_loop_ind = 0
    if mutation_position == 0:
        # creat batch of prefixes of length 1
        start_letters = np.random.choice(["A", "C", "G", "U"], size=batch_size).tolist()
        prefix = tokenizer(start_letters, return_tensors="pt")["input_ids"][:, :-1].to(device)
        start_loop_ind = 1

    # Begin with the prefix and generate n tokens
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(start_loop_ind, mutation_length):
            logits = model(input_ids=prefix).logits[:, -1, :] # shape (batch_size, vocab_size)
            next_tokens, next_scores = sample_next_token(logits, softmax_temperature, logits_threshold, top_k, top_p)
            prefix = torch.cat([prefix, next_tokens.unsqueeze(1)], dim=1).to(device)
            scores[:, i] = next_scores

    # Append the rest of the original sequence (after the mutation)
    suffix = input_data["input_ids"][:, start + mutation_length :]
    mutated_full = torch.cat([prefix, suffix], dim=1)

    mutated_str = tokenizer.batch_decode(mutated_full, skip_special_tokens=True)
    return mutated_str, scores




if __name__ == "__main__":

    
    # set seeds 
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Load the language modelling version of Helix
    model = HelixmRNAForCausalLM.from_pretrained(
    "helical-ai/helix-mRNA",
    attn_implementation="flash_attention_2",
    ).to("cuda")

    tokenizer = CharTokenizer(
        model_max_length=12288,
        padding_side="right",
    )

    # Generate a sequence
    generated_seq = generate(model, tokenizer, max_length_to_generate=50)
    print(generated_seq["sequence"])
    print(generated_seq["scores"])

    # Mutate a sequence of 10 characters
    original = ["ACGUGCAGUC", "GAUCGUACGU"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print("Original sequence:", original)
    print("Mutating sequence in position 0 with length 5...")
    mutated, scores = mutate_sequence(model=model,
                                      device=device,
                                      tokenizer=tokenizer,
                                      mutation_position=0,
                                      original_seq=original,
                                      mutation_length=5,
                                      softmax_temperature=0.6,
                                      top_k=1,
                                      logits_threshold=0.8,
                                      top_p=0.0)
    
    # Compute resulting sequence log likelihoods
    log_likelihoods = compute_sequence_log_likelihood(model, mutated, tokenizer=tokenizer, device=device)
    print("Log likelihoods:", log_likelihoods.cpu().numpy())
    print("Original:", original)
    print("Mutated :", mutated)
    print("Scores  :", scores)

    print("Mutating sequence in position 3 with length 5...")
    mutated, scores = mutate_sequence(model=model,
                                      device=device,
                                      tokenizer=tokenizer,
                                      mutation_position=3,
                                      original_seq=original,
                                      mutation_length=5,
                                      softmax_temperature=0.6,
                                      top_k=1,
                                      logits_threshold=0.8,
                                      top_p=0.0)
    # Compute resulting sequence log likelihoods
    log_likelihoods = compute_sequence_log_likelihood(model, mutated, tokenizer=tokenizer, device=device)
    print("Log likelihoods:", log_likelihoods.cpu().numpy())
    print("Original:", original)
    print("Mutated :", mutated)
    print("Scores  :", scores)

