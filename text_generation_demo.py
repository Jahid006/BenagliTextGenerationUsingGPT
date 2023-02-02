import torch
import numpy as np

from text_processor import Tokenizer
from modeling import GPT
import utils
import config as cfg


def preprocess_text(text, tokenizer, max_len=128):

    input_text = tokenizer.tokenize(text)
    token_len = input_text['token_len']
    input_text = input_text['input_ids']
    input_text = input_text[max(0, (token_len-max_len-2)):]

    input_text = (
        [tokenizer.start_token_id]
        + input_text
        + [tokenizer.end_token_id]
    )
    start_index = len(input_text) - 1
    input_text = (
        (max_len - len(input_text)) * [tokenizer.pad_token_id]
        + input_text
    )
    return input_text, start_index


def do_sample_token(logits, topk=3):
    logits, indices = torch.topk(logits, topk)
    preds = torch.nn.functional.softmax(logits.unsqueeze(0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)


def generate_text(
    model, tokenizer, prefix_text,
    max_seq_len=192,
    max_generated_text=32,
    device=None
):

    tokenized_prefix = tokenizer.tokenize(prefix_text)['input_ids']
    print(f"Length of tokenized Input is {len(tokenized_prefix)}")

    if len(tokenized_prefix) > max_seq_len//2:
        tokenized_prefix = tokenized_prefix[-(max_seq_len//2):]

    num_tokens_generated = 0
    tokens_generated = []
    while num_tokens_generated <= max_generated_text:
        pad_len = max_seq_len - len(tokenized_prefix)
        start_index = len(tokenized_prefix) - 1

        if len(tokenized_prefix) > max_seq_len:
            input_prefix = tokenized_prefix[-max_seq_len:]
            start_index = len(input_prefix) - 1
        
        elif pad_len > 0:
            input_prefix = tokenized_prefix + pad_len*[tokenizer.pad_token_id]
        else:
            input_prefix = tokenized_prefix
        
        logits = prediction_step(model, input_prefix, device)

        sampled_token = do_sample_token(logits[0][start_index])
        tokens_generated.append(int(sampled_token))
        tokenized_prefix.append(sampled_token)

        num_tokens_generated = len(tokens_generated)

    return tokens_generated, tokenizer.tokenizer.decode(tokens_generated)


def prediction_step(model, input_text, device):
    model.eval()
    model.to(device)

    with torch.no_grad():
        input_text = torch.LongTensor([input_text]).to(device)
        output = model(input_text).detach().cpu()

    return output


def predict(text):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Selected Device: {device}')

    vectorizer = Tokenizer(
        sentencepiece_path=cfg.TOKENIZER_PATH,
        max_len=cfg.TOKENIZER_MAX_SEQ,
        vocab_size=cfg.TOKENIZER_VOCAB_SIZE
    )
    vectorizer.load()

    model = GPT(config=cfg.gpt_config).to(device)

    model = utils.load_pretrained_model(
        model=model,
        saved_path=cfg.PRETRAINED_MODEL_PATH,
        device=device
    )

    output = generate_text(
        model=model,
        tokenizer=vectorizer,
        prefix_text=text,
        max_seq_len=cfg.MAX_LEN,
        max_generated_text=64,
        device=device
    )
    
    with open('sample_prediction.txt', 'w') as f:
        f.write(text+'.'*10+output[1])


if __name__ == "__main__":
    text = """
    জেনারেল এরশাদের সিদ্ধান্ত অমান্য করে তাঁর স্ত্রী রওশন এরশাদ যে আওয়ামী লীগের সঙ্গে সমঝোতা করে ফেলেছিলেন, সে কথাও সবার জানা। রওশন এরশাদের সিদ্ধান্তেই সেদিন জাতীয় পার্টির অন্য নেতারা মনোনয়নপত্র জমা দিয়েছেন এবং নির্বাচনে প্রতিদ্বন্দ্বিতা করেছেন।
    জেনারেল এরশাদের নামে যেসব আসনে মনোনয়নপত্র জমা পড়েছিল, সেগুলোর যথার্থতা নিয়েও তখন প্রশ্ন উঠেছিল। 
    """
    predict(text)
