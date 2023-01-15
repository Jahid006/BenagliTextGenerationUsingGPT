
import random
import numpy as np
from itertools import compress
import torch


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, preprocessor, max_len):
        self.dataset = data
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_len = max_len

        print(f"Total {len(data)} Sentence found!!!")

    def _shorten_data(self, dataset_split=.1):
        print('Shortening The Dataset to:', str(100*dataset_split) + '%')

        selected_idx = np.arange(len(self.dataset))
        np.random.shuffle(selected_idx)

        selected_idx = selected_idx[:int(dataset_split*len(selected_idx))]
        self.dataset = list(compress(self.dataset, selected_idx))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        text = self.dataset[index]
        input_text, target_text = self.preprocessor(
            text=text,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

        return {
            "input_text": input_text,
            "target_text": target_text
        }


def text_preprocessor(text, tokenizer, max_len, random_window=False):
    tokenized_text = tokenizer.tokenize(text, padding=False)

    if random_window and tokenized_text['token_len'] > max_len:
        random_index = random.randint(0, tokenized_text['token_len'] - max_len)
        text = tokenized_text['input_ids'][random_index:random_index + max_len]

    text = tokenized_text['input_ids'][:max_len-2]
    input_text = (
        [tokenizer.start_token_id]
        + text
        + [tokenizer.end_token_id]
    )
    input_text = (
        input_text
        + (max_len - len(input_text)) * [tokenizer.pad_token_id]
    )
    target_text = input_text[1:] + [tokenizer.pad_token_id]

    assert len(input_text) == len(target_text),\
        (len(input_text), len(target_text))

    return [torch.LongTensor(input_text), torch.LongTensor(target_text)]


if __name__ == "__main__":
    import config as cfg
    from text_processor import Tokenizer

    sample_text = [
        """অন্যদিকে নামে-বেনামে দেওয়া ঋণ আদায় হচ্ছে না। অনেক ব্যাংক তারল্যঘাটতিতে পড়ছে। 
        ব্যাংকিং খাতে আমানতের প্রবৃদ্ধি প্রায় অর্ধেক নেমেছে। গত অর্থবছরের জুলাই-অক্টোবরের তুলনায়"""
    ]

    sp_tokenizer = Tokenizer(
        sentencepiece_path=cfg.TOKENIZER_PATH,
        max_len=cfg.TOKENIZER_MAX_SEQ,
        vocab_size=cfg.TOKENIZER_VOCAB_SIZE
    )
    sp_tokenizer.load()

    generator = DataGenerator(
        data=sample_text,
        tokenizer=sp_tokenizer,
        preprocessor=text_preprocessor,
        max_len=128
    )
    print(sample_text)
    for data in generator:
        input_text = data['input_text']
        target_text = data['target_text']

        input_text, target_text = map(
            lambda z: [int(i) for i in z.numpy()],
            [input_text, target_text]
        )

        for tokens in [input_text, target_text]:
            t = sp_tokenizer.tokenizer.DecodeIds(tokens)
            print(tokens, t)
