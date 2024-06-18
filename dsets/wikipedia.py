""" 
Parts of the code is based on source code of memit

MIT License

Copyright (c) 2022 Kevin Meng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets import load_dataset


class TokenizedDataset(Dataset):
    """
    Converts a dataset of text samples into a dataset of token sequences,
    as converted by a supplied tokenizer. The tokens come along with position
    ids and attention masks, they can be supplied direcly to the model.
    """

    def __init__(self, text_dataset, tokenizer=None, maxlen=None, field="text"):
        self.text_dataset = text_dataset
        self.field = field
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        if hasattr(text_dataset, "info"):
            self.info = text_dataset.info

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, i):
        text = self.text_dataset[i]
        if self.field is not None:
            text = text[self.field]
        token_list = self.tokenizer.encode(
            text, truncation=True, max_length=self.maxlen
        )
        position_ids = list(range(len(token_list)))
        attention_mask = [1] * len(token_list)       
        return dict(
            input_ids=torch.tensor(token_list).unsqueeze(0),
            position_ids=torch.tensor(position_ids).unsqueeze(0),
            attention_mask=torch.tensor(attention_mask).unsqueeze(0),
        )


def get_ds(tok, ds_name='wikipedia', subset='train', maxlen=1024, batch_tokens=None):
    """ Modiifed function to load wikipedia dataset 
    """
    raw_ds = load_dataset(
        ds_name,
        dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
    )
    if batch_tokens is not None and batch_tokens < maxlen:
        maxlen = batch_tokens
    return raw_ds[subset], TokenizedDataset(raw_ds[subset], tok, maxlen=maxlen)