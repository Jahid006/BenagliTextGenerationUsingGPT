# Bengali Text Generation using GPT


```
Directory Structure
├── artifact                                                             - Artifact directory
│   └── exp-1                                                            - Experiment directory
│       ├── model_step_41600_epoch_099_loss_0.024063_vloss_0.025048.pt   - Saved model
│       ├── config.py                                                    - Experiment config
│       ├── events.out.tfevents                                          - Tensorboard logger
│       └── training.log                                                 - Loggings
├── config.py                                                            - config file
├── data
│   └── opnion_text_512.zip                                              - Text data, unzip it
├── datagen.py                                                           - Dataloader
├── main.py                                                              - Script for Training
├── modeling                                                             - Model Directory
│   └── gpt.py                                                           - GPT model
├── readme.md                                                            - Documentation
├── requirements.txt                                                     - Requirement File
├── sample_prediction.txt
├── text_generation_demo.py                                              - Text Generation Demo
├── text_processor.py                                                    - Tokenizer
├── tokenizer                                                            - Vocabulary Directory
│   └── vocab_complete
│       ├── tokenizer.8000.model
│       └── tokenizer.8000.vocab
├── trainer.py                                                           - Trainer Script
└── utils.py                                                             - Uility Functions
```

## Quick setup

Requires git, python, and conda.

1. Clone this project:
    ```bash
    git clone https://github.com/Jahid006/BnTextGenerationGPT.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. To Train:
    - Update configuration
    - Train a Tokenizer
    ```python
    import config as cfg
    from text_processor import Tokenizer
    sp_tokenizer = Tokenizer(
        sentencepiece_path=cfg.TOKENIZER_PATH,
        max_len=cfg.TOKENIZER_MAX_SEQ,
        vocab_size=cfg.TOKENIZER_VOCAB_SIZE
    )
    sp_tokenizer.train(cfg.DATA_PATH)
    ```
    - Or you can load a pretrained Tokenizer
    ```python
    import config as cfg
    from text_processor import Tokenizer
    sp_tokenizer = Tokenizer(
        sentencepiece_path=cfg.TOKENIZER_PATH,
        max_len=cfg.TOKENIZER_MAX_SEQ,
        vocab_size=cfg.TOKENIZER_VOCAB_SIZE
    )
    sp_tokenizer.load()
    ```
    - Define your Torch Train/Validation Dataset object
    ```python
    import config as cfg
    from datagen import (
        DataGenerator,
        text_preprocessor as preprocessor
    )
    generator = DataGenerator(
        data,
        tokenizer=sp_tokenizer,
        preprocessor=preprocessor,
        max_len=cfg.MAX_LEN,
    )
    ```
    - Define Dataloader object with the Dataset
    - Define Optimizer, Schedular, Criterion
    - Train the model
    ```python
    import config as cfg
    import trainer 
    from torch.utils.tensorboard import SummaryWriter
    model = trainer.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_dataset=train_dataloader,
        test_dataset=val_dataloader,
        device=device,
        epochs=cfg.EPOCHS,
        summary_writter=SummaryWriter(cfg.MODEL_SAVE_DIR),
        logging=logging,
        saving_step=1500,
        steps_per_epoch=len(train_generator),
        model_saving_dir=cfg.MODEL_SAVE_DIR
    )
    ```
    - main.py has a complete walkthrough of training process
4. Follow text_generation_demo.py for inference
5. Sample Prediction
    ```python
    import config as cfg

    text = জেনারেল এরশাদের সিদ্ধান্ত অমান্য করে তাঁর স্ত্রী রওশন এরশাদ যে আওয়ামী লীগের সঙ্গে সমঝোতা করে ফেলেছিলেন, সে কথাও সবার জানা। রওশন এরশাদের সিদ্ধান্তেই সেদিন জাতীয় পার্টির অন্য নেতারা মনোনয়নপত্র জমা দিয়েছেন এবং নির্বাচনে প্রতিদ্বন্দ্বিতা করেছেন। জেনারেল এরশাদের নামে যেসব আসনে মনোনয়নপত্র জমা পড়েছিল, সেগুলোর যথার্থতা নিয়েও তখন প্রশ্ন উঠেছিল।
    output = text_generation_demo.generate_text(
        model=model,
        tokenizer=sp_tokenizer,
        prefix_text=text,
        max_seq_len=cfg.MAX_LEN,
        max_generated_text=64,
        device=device
    )
    output = "এরশাদের পতনের পর বিএনপি-জামায়াতের নেতারা একপর্যায়ে আওয়ামী লীগকে ক্ষমতার অংশী করেছিলেন। এরশাদ ক্ষমতার অংশীদার ছিলেন।"
    ```


## Coming Soon
* Details Documentation
* Function Docstring