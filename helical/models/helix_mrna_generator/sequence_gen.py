from tqdm import trange
from helical.models.helix_mrna_generator.model.modelling_helix_mrna import HelixmRNAForCausalLM
from helical.models.helix_mrna_generator.model.hg38_char_tokenizer import CharTokenizer
import torch
import random

# TODO: unite with regular helix?

def softmax_with_temperature(logits, temperature):
    """
    Apply softmax with temperature scaling to the logits

    Parameters
    ----------
    logits : torch.Tensor
        Logits
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
        Token logits
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
    )
    topk = token_logits_with_temp.topk(top_k)
    top_logits = topk.values
    top_elements = topk.indices

    if (
        token_logits_with_temp[top_elements[0]].item()
        < definite_selection_threshold
    ):
        if top_p > 0:
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
            sampled_index = 0
        else:
            sampled_index = torch.multinomial(top_logits, 1).item()
    else:
        sampled_index = 0

    return (
        top_elements[sampled_index],
        torch.nn.functional.softmax(token_logits, dim=-1)[top_elements[sampled_index]],
    )

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

                new_value, score = sample_next_token(
                    logits,
                    softmax_temperature,
                    definite_selection_threshold,
                    top_k,
                    top_p,
                )

                input_data = append_next_token(input_data, new_value)
                scores = torch.cat([scores, score.unsqueeze(0)])

            sequence = tokenizer.decode(input_data["input_ids"].squeeze())

            generated_sequence["sequence"].append(sequence)
            generated_sequence["scores"].append(scores.cpu().float().numpy())

    return generated_sequence

if __name__ == "__main__":
    # Load the language modelling version of Helix
    model = HelixmRNAForCausalLM.from_pretrained(
    "helical-ai/helix-mRNA",
    attn_implementation="flash_attention_2",
    ).to("cuda")

    tokenizer = CharTokenizer(
        model_max_length=12288,
        padding_side="right",
    )

    generated_seq = generate(model, tokenizer, max_length_to_generate=20)
    print(generated_seq["sequence"])
    print(generated_seq["scores"])