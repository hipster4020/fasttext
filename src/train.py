# -*- coding: utf-8 -*-
import pickle

import pandas as pd
from gensim.models import FastText
from gensim.test.utils import get_tmpfile

import config


# data load
def data_load():
    # pickle load
    with open(config.data_dir, "rb") as f:
        df = pickle.load(f)
    df = pd.DataFrame(df)

    df = df[df.astype(str)["token"] != "[]"]
    tokenized_data = df.token.tolist()

    return tokenized_data


# fasttext
def fasttext(tokenized_data):
    model = FastText(
        sentences=tokenized_data,
        window=config.window,
        min_count=config.min_count,
        workers=config.workers,
        sg=config.sg,
    )

    return model


if __name__ == "__main__":
    tokenized_data = data_load()
    print(tokenized_data[:10])

    # train & save
    model = fasttext(tokenized_data)
    fname = get_tmpfile(config.fname_dir)
    model.save(fname)

    # similar test
    sim_test = model.wv.most_similar("엘지")
    print(sim_test)
