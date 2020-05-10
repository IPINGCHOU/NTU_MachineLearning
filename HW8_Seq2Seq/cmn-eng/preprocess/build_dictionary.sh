#!/bin/bash
python tokenizer.py

subword-nmt learn-bpe -s 5000 < en.txt > en_code.txt
subword-nmt apply-bpe -c en_code.txt < en.txt > en_refine.txt
subword-nmt get-vocab --input en_refine.txt --output en_vocab.txt

python build_dataset.py
