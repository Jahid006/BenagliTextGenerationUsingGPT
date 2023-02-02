import os, shutil, logging, random
from datagen import (
    DataGenerator,
    text_preprocessor as preprocessor
)
import torch
from torch.utils.tensorboard import SummaryWriter

import trainer
from text_processor import Tokenizer
from modeling import GPT
import utils
import config as cfg

random.seed(37)


def main(cfg):

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Selected Device: {device}')

    data = open(cfg.DATA_PATH, 'r').readlines()
    random.shuffle(data)

    # data = data[:int(len(data)*.4)]

    sp_tokenizer = Tokenizer(
        sentencepiece_path=cfg.TOKENIZER_PATH,
        max_len=cfg.TOKENIZER_MAX_SEQ,
        vocab_size=cfg.TOKENIZER_VOCAB_SIZE
    )
    sp_tokenizer.load()

    train_generator = DataGenerator(
        data[:int(len(data)*.95)],
        tokenizer=sp_tokenizer,
        preprocessor=preprocessor,
        max_len=cfg.MAX_LEN,
    )
    val_generator = DataGenerator(
        data[int(len(data)*.95):],
        tokenizer=sp_tokenizer,
        preprocessor=preprocessor,
        max_len=cfg.MAX_LEN,
    )

    sampler = torch.utils.data.RandomSampler(
        train_generator,
        replacement=True,
        num_samples=len(train_generator)//4
    )

    train_generator = torch.utils.data.DataLoader(
        train_generator,
        batch_size=cfg.BATCH_SIZE,
        # shuffle=True,
        prefetch_factor=2,
        num_workers=8,
        sampler=sampler
    )
    val_generator = torch.utils.data.DataLoader(
        val_generator,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        prefetch_factor=2,
        num_workers=4
    )

    print(
        f"Training Dataset Size: {len(train_generator)}"
        + '\n'
        + f"Validation Dataset Size: {len(val_generator)}"
    )

    model = GPT(config=cfg.gpt_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.33,
        patience=5, verbose=True, min_lr=0.000001
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)

    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    shutil.copyfile("config.py", f"{cfg.MODEL_SAVE_DIR}/config.py")

    logging.basicConfig(
        filename=os.path.join(cfg.MODEL_SAVE_DIR, 'training.log'),
        level=logging.DEBUG
    )

    if cfg.LOAD_PRETRAIN:
        model = utils.load_pretrained_model(
            model=model,
            saved_path=cfg.PRETRAINED_MODEL_PATH
        )
        # to do: load optimizer, schedular

    model = trainer.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_dataset=train_generator,
        test_dataset=val_generator,
        device=device,
        epochs=cfg.EPOCHS,
        summary_writter=SummaryWriter(cfg.MODEL_SAVE_DIR),
        logging=logging,
        saving_step=1500,  # 5*len(train_generator)//(cfg.BATCH_SIZE),
        steps_per_epoch=len(train_generator),
        model_saving_dir=cfg.MODEL_SAVE_DIR
    )


if __name__ == "__main__":
    main(cfg)
