DATA_PATH = 'data/text_512.txt'
MODEL_SAVE_DIR = './artifact/exp-2'

LOAD_PRETRAIN = False
PRETRAINED_MODEL_PATH = 'artifact/exp-2/000043056_000008_loss_0.026768_vloss_0.025756.pt'

MAX_LEN = 192
EMBEDDING_DIM = 384
BATCH_SIZE = 96

TOKENIZER_PATH = "./vocab/vocab_complete/tokenizer"
TOKENIZER_MAX_SEQ = 192
TOKENIZER_VOCAB_SIZE = 8000

EPOCHS = 100
INTIAL_EPOCH = 0
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
