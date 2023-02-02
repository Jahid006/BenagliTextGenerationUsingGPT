DATA_PATH = 'data/opnion_text_512.txt'
MODEL_SAVE_DIR = './artifact/exp-3'

LOAD_PRETRAIN = True
PRETRAINED_MODEL_PATH = 'artifact/exp-3/000041600_000099_loss_0.024063_vloss_0.025048.pt'

MAX_LEN = 192
EMBEDDING_DIM = 384
BATCH_SIZE = 96

TOKENIZER_PATH = "./tokenizer/vocab_complete"
TOKENIZER_MAX_SEQ = 192
TOKENIZER_VOCAB_SIZE = 8000

EPOCHS = 100
INTIAL_EPOCH = 12
WORKER = 6
LEARNING_RATE = 0.0005
MULTIPROCESSING = True


class GPTConfig:
    attn_dropout = 0.1
    embed_dropout = 0.1
    ff_dropout = 0.1
    vocab_size = TOKENIZER_VOCAB_SIZE
    max_len = MAX_LEN
    num_heads = 6
    num_blocks = 4
    embed_dim = EMBEDDING_DIM


gpt_config = GPTConfig()
