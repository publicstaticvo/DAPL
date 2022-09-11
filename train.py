import os
import json
import tqdm
import torch
import random
import argparse
import numpy as np
from dataset import Data
from models.bert import BertConfig
from models.dapl import DAPL
from torch.utils.data import DataLoader
from transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def flat(x):
    if len(x.shape) >= 3:
        return x.view(-1, *x.shape[2:])
    else:
        return x.view(-1)


def get_logit(q, s, label=None):
    prototypes = torch.mean(s.view(args.num_ways, args.num_shots, -1), dim=1)
    distance = torch.norm(q.unsqueeze(1) - prototypes.unsqueeze(0), p=2, dim=-1)
    if label is None:
        return torch.mean(cross_entropy(-distance, torch.arange(args.num_ways).to(args.device)))
    else:
        return label[torch.argmin(distance, dim=-1)]


def train(args):
    train_file = os.path.join(args.data_dir, "train.txt")
    dep_file = os.path.join(args.data_dir, "dep_type.json")
    train_data = Data(train_file, args.max_length, n=args.num_ways, k=args.num_shots, model=args.model,
                      dep_file=dep_file, steps=args.train_steps, dep_type=args.dep_type, direct=args.direct)
    print("Finish processing data, there are {} labels".format(len(train_data)))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    config = BertConfig.from_json_file(os.path.join(args.model_path, "config.json"))
    config.__dict__["num_dep"] = 0 if train_data.types_dict is None else len(train_data.types_dict)
    config.__dict__["num_layers"] = args.num_layers
    config.__dict__["leaky_relu"] = args.leakyrelu
    config.__dict__["model"] = args.model
    bert = DAPL.from_pretrained(args.model_path, config=config).to(args.device)
    print("Finished loading pretrained model")
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    decay = [p for n, p in bert.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay = [p for n, p in bert.named_parameters() if any(nd in n for nd in no_decay)]
    ogp = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    optimizer = AdamW(ogp, lr=args.lr, eps=1e-8)
    warmup_steps = int(args.warmup * args.train_steps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=args.train_steps)
    it = train_loader if args.dont_show else tqdm.tqdm(train_loader, desc="Train")
    bert.train()
    for (i, batch) in enumerate(it):
        batch = tuple(flat(t.to(args.device)) for t in batch)
        s_input, s_mask, s_e1_mask, s_e2_mask, s_valid, s_dep_type, s_dp, q_input, q_mask, q_e1_mask, q_e2_mask, q_valid, q_dep_type, q_dp, _ = batch
        if args.model in ["none", "linear"]:
            s_features = bert(s_input, s_e1_mask, s_e2_mask, s_mask)
            q_features = bert(q_input, q_e1_mask, q_e2_mask, q_mask)
        elif args.model in ["gcn", "gat", "agcn"]:
            s_features = bert(s_input, s_e1_mask, s_e2_mask, s_mask, valid=s_valid, dep_matrix=s_dep_type)
            q_features = bert(q_input, q_e1_mask, q_e2_mask, q_mask, valid=q_valid, dep_matrix=q_dep_type)
        else:
            s_features = bert(s_input, s_e1_mask, s_e2_mask, s_mask, dep_pos=s_dp)
            q_features = bert(q_input, q_e1_mask, q_e2_mask, q_mask, dep_pos=q_dp)
        loss = get_logit(q_features, s_features)
        if not args.dont_show:
            it.set_postfix_str(f"loss: {loss:.4f}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if (i + 1) % args.ckpt_steps == 0:
            print(f"acc_on_val:{evaluate_few_shot(bert, args):.4f}")
            print(f"acc_on_test:{evaluate_few_shot(bert, args, mode='test'):.4f}")


def evaluate_few_shot(bert, args, mode="val"):
    test_file = os.path.join(args.data_dir, mode + ".txt")
    dep_file = os.path.join(args.data_dir, "dep_type.json")
    test_data = Data(test_file, args.max_length, n=args.num_ways, k=args.num_shots, direct=args.direct,
                     steps=args.eval_steps, dep_file=dep_file, dep_type=args.dep_type, model=args.model)
    test_loader = DataLoader(test_data, batch_size=1)
    labels = None
    logits = None
    bert.eval()
    for batch in test_loader if args.dont_show else tqdm.tqdm(test_loader, desc="Evaluation"):
        batch = tuple(flat(t.to(args.device)) for t in batch)
        s_input, s_mask, s_e1_mask, s_e2_mask, s_valid, s_dep_type, s_dp, q_input, q_mask, q_e1_mask, q_e2_mask, q_valid, q_dep_type, q_dp, label = batch
        with torch.no_grad():
            if args.model in ["none", "linear"]:
                s_features = bert(s_input, s_e1_mask, s_e2_mask, s_mask)
                q_features = bert(q_input, q_e1_mask, q_e2_mask, q_mask)
            elif args.model in ["gcn", "gat", "agcn"]:
                s_features = bert(s_input, s_e1_mask, s_e2_mask, s_mask, valid=s_valid, dep_matrix=s_dep_type)
                q_features = bert(q_input, q_e1_mask, q_e2_mask, q_mask, valid=q_valid, dep_matrix=q_dep_type)
            else:
                s_features = bert(s_input, s_e1_mask, s_e2_mask, s_mask, dep_pos=s_dp)
                q_features = bert(q_input, q_e1_mask, q_e2_mask, q_mask, dep_pos=q_dp)
            logit = get_logit(q_features, s_features, label)
        if logits is None:
            logits = logit.detach().cpu().numpy()
            labels = label.detach().cpu().numpy()
        else:
            logits = np.concatenate([logits, logit.detach().cpu().numpy()])
            labels = np.concatenate([labels, label.detach().cpu().numpy()])
    return accuracy_score(labels, logits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_steps", default=1000, type=int)
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--dep_type", default="DS", type=str, choices=["None", "D", "S", "DS"])
    parser.add_argument("--direct", action='store_true')
    parser.add_argument("--dont_show", action='store_true')
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument("--leakyrelu", default=0.2, type=float)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--max_length", default=90, type=int)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--model", default="linear", type=str, choices=["none", "linear", "gcn", "gat", "agcn", "dapl"])
    parser.add_argument("--num_ways", default=5, type=int)
    parser.add_argument("--num_shots", default=1, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--train_steps", default=30000, type=int)
    parser.add_argument("--warmup", default=0.01, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cross_entropy = torch.nn.CrossEntropyLoss()
    train(args)
