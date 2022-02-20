from gensim.models.keyedvectors import KeyedVectors
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
import re
import modin.pandas as mp

import config


# fasttext load
def model_load():
    fname = get_tmpfile(config.fname_dir)
    fasttext = FastText.load(fname)
    
    return fasttext


# company name load
def data_load():
    # company_name
    colnames=['company_name'] 
    company_name = mp.read_csv(
        config.company_name, names=colnames, sep="\n",header=None)
    
    print(len(company_name))
    return company_name


# similar predict
def predict(data):
    sim_words = fasttext.wv.most_similar(data)
    sim_list = [sim[0] for sim in sim_words if sim[1] >= 0.8]

    return sim_list


# synonym save
def synonym_save(total_list):
    with open(config.synonym_dir, "w") as f:
        for i in total_list:
            f.write(",".join(i) + "\n")


if __name__ == "__main__":
    fasttext = model_load()
    data = data_load()
    
    # predict
    data['fasttext_org'] = data.company_name.apply(predict)
    data['fasttext'] = data.fasttext_org.apply(lambda x : ','.join(x))
    
    # save
    total_list = data.fasttext.to_list()
    synonym_save(total_list)
