import torch
from torch.utils.data import Dataset
from gensim.utils import simple_preprocess

class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=120):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.id2label = {0: "NEG", 1: "POS", 2:"NEU"}
        self.label2id = {"NEG": 0, "POS": 1, "NEU": 2}
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text, label = self.get_input_data(row)
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_masks': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long),
        }

    def get_input_data(self, row):
        # Preprocessing:w{remove icon, special character, lower}
        text = row['wseg']
        # text = ' '.join(simple_preprocess(text))
        label = self.label2id[row['label']]
        return text, label
