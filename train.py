import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils import data

import argparse

from loader import DataLoader, pad, load_vocab
from model import Net

import os

def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        x, x_seqlens, sources, y, y_seqlens, targets, is_heads = batch

        optimizer.zero_grad()

        logits, y, y_hat = model(x, y)

        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i and i % 10==0:
            print(f'step: {i}, loss: {loss.item()}')

def calc_score(true_list, pred_list, tag2idx):
    y_true = np.array([tag2idx[each] for each in true_list])
    y_pred = np.array([tag2idx[each] for each in pred_list])

    num_proposed = len(y_pred[y_pred > 3])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 3)).astype(int).sum()
    num_gold = len(y_true[y_true > 3])

    try:
        pre = num_correct / num_proposed
    except ZeroDivisionError:
        pre = 1.0
    try:
        re = num_correct / num_gold
    except ZeroDivisionError:
        re = 1.0
    try:
        f1 = 2*pre*re / (pre + re)
    except ZeroDivisionError:
        f1 = 1.0

    print(f'---- evaluation result ----')
    print(f'precision : {pre}')
    print(f'recall : {re}')
    print(f'f1 score : {f1}')

    return pre, re, f1


def eval(model, iterator, epoch, tag2idx, idx2tag):
    model.eval()

    Words, Targets, Y, Y_hat, Is_heads = [], [], [], [], []
    pred_list, true_list = [], []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, x_seqlens, _, y, y_seqlens, targets, is_heads = batch

            _, _, y_hat = model(x, y)

            Targets.extend(targets)
            Is_heads.extend(is_heads)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    for targets, y_hat, is_heads in zip(Targets, Y_hat, Is_heads):
        preds = [idx2tag[hat] for hat, head in zip(y_hat, is_heads) if head==1]
        for t, p in zip(targets[1:-1], preds[1:-1]):
            pred_list.append(p)
            true_list.append(t)

    pre, re, f1 = calc_score(true_list, pred_list, tag2idx)

    return pre, re, f1

def pred(model, iterator, tag2idx, idx2tag):
    model.eval()

    Words, Targets, Y_hat, Is_heads = [], [], [], []
    true_list, pred_list = [], []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, x_seqlens, sources, y, y_seqlens, targets, is_heads = batch

            _, _, y_hat = model(x, y)

            Words.extend(sources)
            Targets.extend(targets)
            Is_heads.extend(is_heads)
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    with open('result.txt', 'w') as fout:
        for x, targets, y_hat, is_heads in zip(Words, Targets, Y_hat, Is_heads):
            preds = [idx2tag[hat] for hat, head in zip(y_hat, is_heads) if head==1]
            for w, t, p in zip(x[1:-1], targets[1:-1], preds[1:-1]):
                true_list.append(t)
                pred_list.append(p)
                fout.write(f'{w} {t} {p}\n')
            fout.write('\n')

    _, _, _ = calc_score(true_list, pred_list, tag2idx)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--early_stopping_step", type=int, default=15)
    parser.add_argument("--logdir", type=str, default="checkpoints")
    parser.add_argument("--trainset", type=str, default="data/train.txt")
    parser.add_argument("--testset", type=str, default="data/test.txt")
    parser.add_argument("--validset", type=str, default="data/val.txt")
    parser.add_argument("--model_path", type=str, default="checkpoints/129.pt")
    parser.add_argument("--vocab_path", type=str, default="vocab.txt")
    parser.add_argument("--finetuning", dest="finetuning", action='store_true')
    parser.add_argument("--train", dest="train", action='store_true')
    parser.add_argument("--pred", dest="pred", action='store_true')

    hp = parser.parse_args()

    if not os.path.isdir(hp.logdir):
        os.mkdir(hp.logdir)
    if hp.train:
        hp.finetuning = True
        train_dataset = DataLoader(hp.trainset, hp)
        train_iter = data.DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=pad)
        eval_dataset = DataLoader(hp.validset, hp)
        eval_iter = data.DataLoader(eval_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=pad)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(device=device, hidden_size=hp.hidden_size, finetuning=hp.finetuning, tag_size=len(train_dataset.vocab))
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=hp.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        maxf1 = 0.0
        step = 0

        for epoch in range(1, hp.n_epochs+1):
            print(f'train in {epoch}...')
            train(model, train_iter, optimizer, criterion)
            pre, re, f1 = eval(model, eval_iter, epoch, eval_dataset.tag2idx, eval_dataset.idx2tag)

            if maxf1 < f1:
                maxf1 = f1
                if os.path.exists(hp.model_path):
                    os.remove(hp.model_path)
                torch.save(model, f'{hp.logdir}/{str(epoch)}.pt')
                hp.model_path = f'{hp.logdir}/{str(epoch)}.pt'
                step = 0

            if step > hp.early_stopping_step:
                print(f'early stopping on {epoch}')
                break
            else:
                step += 1
    if hp.pred:
        pred_dataset = DataLoader(hp.testset, hp)
        pred_iter = data.DataLoader(pred_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=pad)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(device=device, hidden_size=hp.hidden_size, finetuning=False, tag_size=len(pred_dataset.vocab))
        model.to(device)

        model = torch.load(hp.model_path)
        print(f'load model: {hp.model_path}')

        pred(model, pred_iter, pred_dataset.tag2idx, pred_dataset.idx2tag)
