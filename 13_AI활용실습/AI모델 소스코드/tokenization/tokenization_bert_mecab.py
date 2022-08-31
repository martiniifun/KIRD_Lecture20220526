# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

import collections
import logging
import unicodedata
import six
import os
from shutil import copyfile
from typing import List

from transformers import BertTokenizer, WordpieceTokenizer


logger = logging.getLogger(__name__)

try:
    from konlpy.tag import Mecab
except:
    logger.warning("Could not find Mecab installation! (ignore this if you are not using mecab)")


def convert_to_unicode(text):
    raise ValueError("Not impletemented yet!")


def printable_text(text):
    raise ValueError("Not impletemented yet!")


def load_vocab(vocab_file):
    raise ValueError("Not impletemented yet!")


def convert_by_vocab(vocab, items):
    raise ValueError("Not impletemented yet!")


def convert_tokens_to_ids(vocab, tokens):
    raise ValueError("Not impletemented yet!")


def convert_ids_to_tokens(inv_vocab, ids):
    raise ValueError("Not impletemented yet!")


def whitespace_tokenize(text):
    raise ValueError("Not impletemented yet!")


def strip_accents(text):
    raise ValueError("Not impletemented yet!")


def _is_whitespace(char):
    raise ValueError("Not impletemented yet!")


def _is_control(char):
    raise ValueError("Not impletemented yet!")


def _is_punctuation(char):
    raise ValueError("Not impletemented yet!")


class BertKoreanMecabTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        bos_token="<S>",
        eos_token="[EOS]",
        **kwargs
    ):
        raise ValueError("Not impletemented yet!")

    def tokenize(self, text, mode='full', basic_tokens=None, **kwargs):
        raise ValueError("Not impletemented yet!")

    def detokenize(self, token_list):
        raise ValueError("Not impletemented yet!")

    def strip_accents(self, text):
        raise ValueError("Not impletemented yet!")

    def save_vocabulary(self, vocab_path):
        raise ValueError("Not impletemented yet!")


class BasicTokenizer:
    def __init__(self):
        raise ValueError("Not impletemented yet!")

    def tokenize(self, text, pos_length=2, spacing=False, joining=False):
        raise ValueError("Not impletemented yet!")

    def _run_split_on_punc(self, text):
        raise ValueError("Not impletemented yet!")

    def _tokenize_chinese_chars(self, text):
        raise ValueError("Not impletemented yet!")

    def _is_chinese_char(self, cp):
        raise ValueError("Not impletemented yet!")

    def _clean_text(self, text):
        raise ValueError("Not impletemented yet!")
