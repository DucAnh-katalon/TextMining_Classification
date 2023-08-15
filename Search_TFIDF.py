from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


pwd = Path(os.getcwd())
save_dir = pwd / 'models'
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(save_dir))

data = pd.read_csv('data - data.csv', usecols=range(0, 3))
data['wseg'] = data['comment'].apply(lambda x : " ".join(simple_preprocess( " ".join(rdrsegmenter.word_segment(x)))))

