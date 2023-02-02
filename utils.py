import numpy as np
import torch
from math import log


def preprocess_text(text, tokenizer, max_len=128):

    input_text = tokenizer.tokenize(text)['input_ids'][-max_len:]
    input_text = input_text[:max_len-2]

    input_text = (
        [tokenizer.start_token_id]
        + input_text
        + [tokenizer.end_token_id]
    )
    input_text = (
        (max_len - len(input_text)) * [tokenizer.pad_token_id]
        + input_text
    )
    return np.array(input_text, dtype=np.int32)


def beam_search_decoder(prob_matrix, k):
    sequences = [[list(), 1.0]]
    for row in prob_matrix:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j]+.00001)]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:k]
    return sequences


def load_pretrained_model(model, saved_path, device='cuda', key='model'):
    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(saved_path, map_location=device)
    loaded_state_dict = loaded_state_dict[key]

    new_state_dict = {
        k: v
        if v.size() == current_model_dict[k].size()
        else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
    }

    mis_matched_layers = [
        k for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
        if v.size() != current_model_dict[k].size()
    ]

    if mis_matched_layers:
        print(f"{len(mis_matched_layers)} Mismatched layers found.")
        print(mis_matched_layers)
   
    model.load_state_dict(new_state_dict, strict=True)
    print('model loaded successfully')

    return model