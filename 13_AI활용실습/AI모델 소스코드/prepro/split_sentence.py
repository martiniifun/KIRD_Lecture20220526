import os
import re
import time
import json
import glob
import argparse
import logging
import random
from joblib import Parallel, delayed

import tqdm
from kss import split_sentences

from others.logging import init_logger

def text_processing(text):
    # 반복되는 문자 처리
    repeats = [
        '.', ','
        , "'", '"', "~", "@", "#", "$", "%", "^"
        , "&", "*", "-", "_", "=", "+"
        , "(", ")", "[", "]", "{", "}", "<", ">"
        , "?", "!"
    ]

    for repeat in repeats:
        text = re.sub('\{}+'.format(repeat), repeat, text)

    # 따옴표 처리
    text = text.replace("`", "'").replace("‘", "'").replace("’", "'").replace('“', '"').replace('”', '"')

    # 줄문자 처리
    text = text.replace('\\n', '').replace('\\t', '').replace('\\r', '')

    # 일부 문자 처리
    text = text.replace("∼", "~")

    # 일부 문자 & 한자만 남김.
    text = re.sub('[^ .,?!/@$%~％·∼()\x00-\x7F가-힣\u4e00-\u9fff]+', '', text)

    return text.strip()


def _get_files(path, pattern='*.json'):
    files = sorted(glob.glob(os.path.join(path, pattern)))
    return files


def _read_json_file(file):
    with open(file) as f:
        data = json.load(f)

    return data


def _format_data(data, data_type):
    if data_type == "report":
        # Add metadata
        data["meta"] = data["Meta(Acqusition)"]
        del data["Meta(Acqusition)"]

        data["meta"]["passage_id"] = data["Meta(Refine)"]["passage_id"]
        data["meta"]["passage"] = data["Meta(Refine)"]["passage"]
        data["meta"]["passage_Cnt"] = data["Meta(Refine)"]["passage_Cnt"]
        del data["Meta(Refine)"]

        # Add summary
        if data["meta"]["passage_Cnt"] < 1000:
            data["src"] = data["meta"]["passage"]
            data["src_from"] = "passage"
        elif data["Annotation"]["summary3"]:
            # 20% 요약
            data["src"] = data["Annotation"]["summary3"]
            data["src_from"] = "summary3"
        else:
            # 2~3문장요약
            data["src"] = data["Annotation"]["summary2"]
            data["src_from"] = "summary2"
        data["tgt"] = data["Annotation"]["summary1"]
        del data["Annotation"]

        return [data]


def _get_data(file, data_type):
    logger.info(" Reading file: {}".format(file))

    # read json
    data = _read_json_file(file)

    # format data
    data = _format_data(data, data_type)
    return data


def _split_datasets(data_list, split_datasets, chunk_size):
    train_p, valid_p, test_p = [int(p.strip()) for p in split_datasets.split(",")]
    assert train_p + valid_p + test_p == 1, "sum of split_datasets have to be 1"

    # flat & shuffle data
    data = sum(data_list, [])
    random.shuffle(data)

    # split to train / valid / test
    train = data[:int(len(data)*train_p)] if train_p else []
    valid = data[int(len(data)*train_p):int(len(data)*(train_p+valid_p))] if valid_p else []
    test = data[int(len(data)*(train_p+valid_p)):] if test_p else []

    # split by chunk_size
    train = [train[i:i + chunk_size] for i in range(0, len(train), chunk_size)]
    valid = [valid[i:i + chunk_size] for i in range(0, len(valid), chunk_size)]
    test = [test[i:i + chunk_size] for i in range(0, len(test), chunk_size)]

    return train, valid, test


def _split_chunk(data, chunk_size):
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    return chunks


def _split_sentence(chunk, save_path, save_pattern, index):
    logger.info(" Splitting sentence: {} chunks".format(len(chunk)))
    for row in chunk:
        # clean data
        if args.clean_data:
            row["src"] = text_processing(row["src"])
            row["tgt"] = text_processing(row["tgt"])

        # split sentences & split tokens by spaces
        row["src"] = [sent.split() for sent in split_sentences(row["src"])]
        row["tgt"] = [sent.split() for sent in split_sentences(row["tgt"])]

    logger.info(" Saving file: {}".format(save_pattern.format(index)))
    with open(os.path.join(save_path, save_pattern.format(index)), "w") as f:
        json.dump(chunk, f, indent=4, ensure_ascii=False)


def main(args):
    now = time.time()
    files = _get_files(args.raw_path)

    data_list = Parallel(n_jobs=args.n_cpus, backend="multiprocessing")(
        delayed(_get_data)(file, args.data_type) for file in tqdm.tqdm(files)
    )

    data = sum(data_list, [])
    random.shuffle(data)
    # print(f'len(data): {len(data)}, len(data_list): {len(data_list)}')
    # print(f'data_list: {data_list}')
    # print(f'data: {data}')

    chunks = _split_chunk(data, args.chunk_size)

    Parallel(n_jobs=args.n_cpus, backend="multiprocessing")(
        delayed(_split_sentence)(chunk, args.save_path, args.save_pattern, i) for i, chunk in tqdm.tqdm(enumerate(chunks))
    )

    end = time.time() - now
    logger.info(" Time: {}secs".format(round(end), len(files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-raw_path", default='../../sample_data')
    parser.add_argument("-save_path", default='../../data/')
    parser.add_argument("-save_pattern", default='mecab.{}.train.json')
    parser.add_argument('-log_file', default='../../logs/split_sentence.log')

    parser.add_argument('-n_cpus', default=8, type=int)

    parser.add_argument('-clean_data', default=True, type=str)

    parser.add_argument('-data_type', default='books', type=str, choices=['report', 'broadcast'])
    parser.add_argument('-chunk_size', default=1000, type=int)

    args = parser.parse_args()
    logger = init_logger(args.log_file)
    main(args)
