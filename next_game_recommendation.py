# -*- coding: utf-8 -*-


!pip install transformers
!pip install beautifulsoup4
!pip install lxml
import sys
import json
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import time
import pandas as pd
from bs4 import BeautifulSoup


from google.colab import drive
drive.mount('/content/drive')
#df = pd.read_csv("/content/drive/My Drive/Workspace/app_label_prediction/train_set.csv")
#test_df = pd.read_csv("/content/drive/My Drive/Workspace/app_label_prediction/test_set.csv")

TRAIN_PATH = "/content/drive/My Drive/Workspace/app_label_prediction/train_set.csv" # path to train file
TEST_PATH = "/content/drive/My Drive/Workspace/app_label_prediction/test_set.csv" # path to test file
SAVE_PATH = "/content/drive/My Drive/Workspace/app_label_prediction/models/" # path to save model
LOAD_PATH = "" # path to load model for continuing training


PRE_TRAINED_MODEL = 'HooshvareLab/bert-fa-base-uncased'
MAXTOKENS = 128
NUM_EPOCHS = 30
BERT_EMB = 768
H = 512
BS = 32
INITIAL_LR = 1e-6
WARMUP = 500
# model at the end of the following epochs will be saved
save_epochs = [5]

CUDA_0 = 'cuda:0'
CUDA_1 = 'cuda:0'
CUDA_2 = 'cuda:0'


def myprint(mystr, logfile):
    print(mystr)
    print(mystr, file=logfile)


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def separate_texts_labels(df, isTest=False):
    texts = df['description_fa'].values.tolist()
    for i in range(len(texts)):
        texts[i] = BeautifulSoup(texts[i], "lxml").text
    if not isTest:
        labels = df['label'].values.astype(int).tolist()
        return texts, labels
    else:
        return texts


def load_data(f_train, f_test):
  try:
    trainfile = pd.read_csv(f_train, sep=',', names=['app_id', 'description_fa', 'label'])
    trainfile = trainfile.drop(index=0)
    test_df = pd.read_csv(f_test, sep=',', names=['app_id', 'description_fa'])
    test_df = test_df.drop(index=0)
  except:
    print('my log: could not read file')
    exit()
    # print(trainfile['label'].value_counts())
  else:
    err_printed = False
    for i in range(1, len(trainfile)):
        if not is_integer(trainfile.loc[i, 'label']):
            if not err_printed:
                print('The labels following data rows have problems being read in ', trainfile)
                err_printed = True
            print(trainfile.loc[i, 'label'])
    val_df = trainfile.sample(frac=0.1, random_state=42, replace=False)
    train_df = trainfile.drop(val_df.index)
    # test_df = test_df.sample(frac=1, random_state=42, replace=False)
    train_texts, train_labels = separate_texts_labels(train_df)
    val_texts, val_labels = separate_texts_labels(val_df)
    test_texts = separate_texts_labels(test_df, isTest=True)
    test_ids = test_df['app_id'].values.tolist()
    #tmp = set(trainfile['label'].values)
    #print("uniques: ", len(tmp))
    #print(tmp)
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_ids


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def evaluate_model(labels, predictions, titlestr, logfile):
    myprint(titlestr, logfile)
    conf_matrix = confusion_matrix(labels, predictions)
    myprint("Confusion matrix- \n" + str(conf_matrix), logfile)
    acc_score = accuracy_score(labels, predictions)
    myprint('  Accuracy Score: {0:.2f}'.format(acc_score), logfile)
    myprint('Report', logfile)
    cls_rep = classification_report(labels, predictions)
    myprint(cls_rep, logfile)


def feed_model(model, data_loader, isLabelsKnown):
    outputs_flat = []
    if isLabelsKnown:
        labels_flat = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(CUDA_0)
        attention_mask = batch['attention_mask'].to(CUDA_0)
        outputs = model(input_ids, attention_mask=attention_mask)
        outputs = outputs.detach().cpu().numpy()
        if isLabelsKnown:
            labels = batch['labels'].to('cpu').numpy()
        outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
        if isLabelsKnown:
            labels_flat.extend(labels.flatten())
        del outputs, attention_mask, input_ids
        if isLabelsKnown:
            del labels
    if not isLabelsKnown:
        return outputs_flat
    else:
        return labels_flat, outputs_flat


class MyInternetModel(nn.Module):
    def __init__(self, base_model, n_classes, dropout=0.1):
        super().__init__()

        self.base_model = base_model.to(CUDA_0)
        '''
        self.mylstm = nn.Sequential(
            nn.LSTM(input_size=BERT_EMB, hidden_size=BERT_EMB, num_layers=2, dropout=0.25, batch_first=True,
                    bidirectional=True)
        ).to('cuda:6')
        '''
        self.final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(BERT_EMB, BERT_EMB),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(BERT_EMB, BERT_EMB//2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(BERT_EMB//2, n_classes)
        ).to(CUDA_1)

    def forward(self, input_, **kwargs):
        X = input_
        if 'attention_mask' in kwargs:
            attention_mask = kwargs['attention_mask']
        else:
            print("my err: attention mask is not set, error maybe")
        hidden_states = self.base_model(X.to(CUDA_0), attention_mask=attention_mask.to(CUDA_0)).last_hidden_state
        cls = hidden_states[:, 0, :]
        myo = self.final(cls.to(CUDA_1))
        myo = nn.functional.softmax(myo, dim=1)
        return myo


if __name__ == '__main__':
    args = sys.argv
    epochs = NUM_EPOCHS
    train_texts, train_labels, val_texts, val_labels, test_texts, test_ids = load_data(TRAIN_PATH, TEST_PATH)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    tokenizer.model_max_length = MAXTOKENS
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    train_dataset = MyDataset(train_encodings, train_labels)
    val_dataset = MyDataset(val_encodings, val_labels)
    test_dataset = MyDataset(test_encodings)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logfile = open('log_file_' + args[0].split('/')[-1][:-3] + str(time.time()) + '.txt', 'w')
    #myprint(INITIAL_LR, logfile)

    base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL)
    model = MyInternetModel(base_model=base_model, n_classes=10)
    #model.load_state_dict(torch.load(LOAD_PATH))
    model.train()

    # train and test set are shuffled once in loading, to be reproducible, shuffle=False here
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    optim = AdamW(model.parameters(), lr=INITIAL_LR)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=WARMUP,
                                                num_training_steps=total_steps)
    # since data is imbalanced and every class has double the number of samples of class 5, weights are assigned as
    # below
    class_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5]
    class_weights = torch.FloatTensor(class_weights).to(CUDA_1)
    loss_model = nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(epochs):
        print(' EPOCH {:} / {:}'.format(epoch+1, epochs))
        outputs_flat = []
        labels_flat = []
        for step, batch in enumerate(train_loader):
            if step % 100 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))
            optim.zero_grad()
            input_ids = batch['input_ids'].to(CUDA_0)
            attention_mask = batch['attention_mask'].to(CUDA_0)
            labels = batch['labels'].to(CUDA_1)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_model(outputs, labels)
            loss.backward()
            optim.step()
            scheduler.step()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
            labels_flat.extend(labels.flatten())
            del outputs, labels, attention_mask, input_ids
        evaluate_model(labels_flat, outputs_flat, 'Train set Result epoch ' + str(epoch+1), logfile)
        del labels_flat, outputs_flat
        model.eval()
        val_labels, val_predictions = feed_model(model, val_loader, True)
        evaluate_model(val_labels, val_predictions, 'Validation set Result epoch ' + str(epoch+1), logfile)
        del val_labels, val_predictions

        if (epoch+1) in save_epochs:
            test_predictions = feed_model(model, test_loader, False)
            # evaluate_model(test_labels, test_predictions, 'Test set Result epoch ' + str(epoch+1), logfile)
            preds_df = pd.DataFrame(list(zip(test_ids, test_predictions)), columns=['app_id', 'label'])
            try:
                preds_df.to_csv('prediction' + args[0].split('/')[-1][:-3] + '.csv', index=False)
            except:
                myprint("Prediction file did not save due to error", logfile)
            # save predictions
            del test_predictions
            try:
                torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '-auto-' + str(epoch+1)))
            except:
                myprint("Could not save the model", logfile)

        model.train()
    del train_loader
    model.eval()
    myprint('--------------Training complete--------------', logfile)
    torch.save(model.state_dict(), SAVE_PATH + args[0].split('/')[-1][:-3] + '-final')
    # test_labels, test_predictions = feed_model(model, test_loader)
    # evaluate_model(test_labels, test_predictions, 'Final Testing', logfile)
    # del test_labels, test_predictions
