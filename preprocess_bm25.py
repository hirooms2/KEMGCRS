from rank_bm25 import BM25Okapi
import pickle
import os
from tqdm import tqdm
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, GPT2Config
from datetime import datetime
import argparse
import torch
from itertools import chain
import random

HOME = os.path.dirname(os.path.realpath(__file__))
VERSION = 2
DATA_DIR = os.path.join(HOME, 'data', str(VERSION))
BERT_NAME = 'bert-base-uncased'
CACHE_DIR = os.path.join(HOME, "model_cache", BERT_NAME)
stop_words = set(stopwords.words('english'))
word_piece_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def custom_tokenizer(text):
    text = " ".join([word for word in text.split() if word not in stop_words])
    tokens = word_piece_tokenizer.encode(text)[1:-1]
    return tokens


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def readDic(filename, out=None):
    output_idx_str = dict()
    output_idx_int = dict()
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                k, idx = line.strip().split('\t')
            except:
                print(line)
                k, idx = line.strip().split()
            output_idx_str[k] = int(idx)
            output_idx_int[int(idx)] = k
    if out == 'str':
        return output_idx_str
    elif out == 'idx':
        return output_idx_int
    else:
        return {'str': output_idx_str, 'int': output_idx_int}


def clean_know_texts(know):
    output = []
    output.append(clean_know_text(know[0]))
    output.append(clean_know_text(know[1]))
    output.append(clean_know_text(know[2]))
    return output


def clean_know_text(text):
    output = text.replace('℃', ' degrees Celsius')
    return output


def clean_join_triple(know):
    if isinstance(know, list) and len(know) > 0:
        know = clean_know_texts(know)
        if know[1] == 'Sings':
            return ' '.join([know[0], 'singer', know[2]])
        elif know[1] == 'Stars':
            return ' '.join([know[0], 'star', know[2]])
        elif know[1] == 'Intro':
            return ' '.join([know[0], 'is', know[2]])
        elif know[1] == 'Birthday':
            return ' '.join([know[0], know[1], datetime.strptime(know[2].replace(' ', ''), '%Y-%m-%d').strftime('%Y %B %dth')])
        else:
            return ' '.join(know)
    else:
        return ""


def make(args, mode, dialogs, start=0, end=0, m=None):
    cnt = 0
    filtered_corpus = args.train_know_tokens if mode == 'train' else args.all_know_tokens
    bm25 = BM25Okapi(filtered_corpus)

    # print(mode)
    corpus = list(args.all_knowledges)
    dataset_psd = []
    # with open(f'{HOME}/data/2/en_{mode}.txt', 'r', encoding='UTF-8') as f:
    for index in tqdm(range(start, end), desc=f"{mode.upper()}_Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
        cnt += 1
        dialog = dialogs[index]
        # if cnt==20: break
        # dialog = json.loads(line)
        dialog['know_candidates'] = []
        conversation, knowledge_seq = dialog['conversation'], dialog['knowledge']
        topic_seq, goal_seq = dialog['goal_topic_list'], dialog['goal_type_list']

        role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]
        for i in range(2, len(conversation)):
            role_seq.append(role_seq[i % 2])

        for i in range(len(conversation)):
            conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]

        uidx = -1
        prev_topic = ''
        for (goal, role, utt, know, topic) in zip(goal_seq, role_seq, conversation, knowledge_seq, topic_seq):
            uidx += 1
            response=""
            if 'resp' in args.how: response += utt
            if 'uttr' in args.how and uidx>0: response = conversation[uidx - 1] + response
            
            # if uidx > 0: response = conversation[uidx - 1] + utt
            # else: response = utt

            if goal == 'Food recommendation': response = ' '.join(conversation[:uidx]) + utt
            response = response.replace('℃', ' degrees Celsius')

            response = word_piece_tokenizer.decode(word_piece_tokenizer.encode(response)[1:-1])
            if 'item' in args.how:
                if prev_topic != topic: response = prev_topic + "|" + topic + "|" + response
                else: response = topic + "|" + response

            if know:
                # know = clean_know_texts(know)
                know = clean_join_triple(know)
                # know_idx = corpus.index(know)
                # response_knowledge.append((response.lower(), know.lower(), know_idx, goal, topic, prev_topic))

                tokenized_query = custom_tokenizer(response.lower())
                doc_scores = bm25.get_scores(tokenized_query)

                # doc_scores_tensor = torch.Tensor(doc_scores).to('cuda')
                doc_scores = np.array(doc_scores)

                sorted_rank = doc_scores.argsort()[::-1]
                # sorted_rank_tensor = doc_scores_tensor.argsort(descending=False)
                top1000_retrieved = [corpus[idx] for idx in sorted_rank[:1000]]
                # top1000_retrieved = [corpus[idx] for idx in sorted_rank_tensor[:1000]]
                for rank in range(len(top1000_retrieved)):
                    if topic not in top1000_retrieved[rank]:
                        doc_scores[sorted_rank[rank]] = -1
                        # doc_scores[sorted_rank_tensor[rank]] = -1
                re_sorted_rank = doc_scores.argsort()[::-1]
                # re_sorted_rank_tensor = doc_scores_tensor.argsort(descending=True)
                # prob = softmax(doc_scores)

                candidates_positive_triple = [args.all_knowledges[corpus[idx]] for idx in re_sorted_rank[:20]]
                canditates_postivie_probs = [doc_scores[idx] for idx in re_sorted_rank[:20]]

                # candidates_positive_triple = [args.all_knowledges[corpus[idx]] for idx in re_sorted_rank_tensor[:20]]
                # canditates_postivie_probs = [doc_scores[idx] for idx in re_sorted_rank_tensor[:20]]

                know_candidates = []
                for idx, (tokens, prob) in enumerate(zip(candidates_positive_triple, canditates_postivie_probs)):
                    know_candidates.append((tokens, prob))
                dialog["know_candidates"].append(know_candidates)
            else:
                dialog["know_candidates"].append([])
            prev_topic = topic
        dataset_psd.append(dialog)
        # if m: m.append([index, dialog])
        if m: m.put([index, dialog])
    # eval(dataset_psd)
    # if args.save: save(mode, dataset_psd)
    return dataset_psd


def eval(dataset_psd):
    GOAL_LIST = ['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'Food recommendation']
    hitDic = {f'Top{i}_hit': 0 for i in range(1, 11)}
    hitDic['count'] = 0
    for dialog in dataset_psd:
        knows = [clean_join_triple(i) for i in dialog['knowledge']]
        know_cands = [[clean_join_triple(j[0]) for j in i] for i in dialog['know_candidates']]
        # know_cands = [clean_join_triple(i) for i in dialog['know_candidates']]
        role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]
        for i in range(2, len(dialog['goal_type_list'])):
            role_seq.append(role_seq[i % 2])
        for know, know_cand, goal, role in zip(knows, know_cands, dialog['goal_type_list'], role_seq):
            if know and goal in GOAL_LIST and role == 'System' and len(know_cand) > 0:  # 3711
                hitDic['count'] += 1
                for i in range(10):
                    if know_cand[i] == know:
                        hitDic[f'Top{i + 1}_hit'] += 1
                        break
    for i in range(1, 11):
        print(f"Top{i}_hit_ratio: {hitDic[f'Top{i}_hit'] / hitDic['count'] :.3f}")
    print(f"Total count: {hitDic['count']}")


def save(dataset_psd, mode):
    with open(f'{HOME}/data/2/en_{mode}_know_cand_score20_new.txt', 'a', encoding='utf8') as fw:
        for dialog in dataset_psd:
            fw.write(json.dumps(dialog) + "\n")


def default_parser(parser):
    # Default For All
    parser.add_argument("--version", default='2', type=str, help="Choose the task")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="BERT Model Name")
    parser.add_argument('--mode', default='train', type=str, help="Train/dev/test")
    parser.add_argument('--home', default=os.path.dirname(os.path.realpath(__file__)), type=str, help="Home path")
    parser.add_argument("--save", action='store_true', help="Whether to SAVE")
    parser.add_argument('--how', default='resp_uttr_item', type=str, help="resp_utt_item ablation")
    return parser


if __name__ == "__main__":
    import multiprocessing

    set_seed()

    args = default_parser(argparse.ArgumentParser(description="ours_main.py")).parse_args()
    args.home = os.path.dirname(os.path.realpath(__file__))

    all_knowledges, train_knowledges, valid_knowledges, test_knowledges = dict(), dict(), dict(), dict()
    train_dialogs, dev_dialogs, test_dialogs = list(), list(), list()
    with open(f'{HOME}/data/2/en_train.txt', 'r', encoding='utf8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            train_dialogs.append(dialog)
            for know in dialog['knowledge']:
                if know: train_knowledges[clean_join_triple(know)] = know

    with open(f'{HOME}/data/2/en_dev.txt', 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            dev_dialogs.append(dialog)
            for know in dialog['knowledge']:
                if know: valid_knowledges[clean_join_triple(know)] = know

    with open(f'{HOME}/data/2/en_test.txt', 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            dialog = json.loads(line)
            test_dialogs.append(dialog)
            for know in dialog['knowledge']:
                if know: test_knowledges[clean_join_triple(know)] = know

    filtered_corpus_train = []
    filtered_corpus_all = []
    for knows in [train_knowledges, valid_knowledges, test_knowledges]:
        for k, v in knows.items():
            all_knowledges[k] = v

    args.train_knowledges = train_knowledges
    args.all_knowledges = all_knowledges

    for sent in tqdm(args.train_knowledges, desc="Train_know_tokenize", bar_format='{l_bar} | {bar:23} {r_bar}'):
        filtered_corpus_train.append(custom_tokenizer(sent))
    for sent in tqdm(args.all_knowledges, desc="Test_know_tokenize", bar_format='{l_bar} | {bar:23} {r_bar}'):
        filtered_corpus_all.append(custom_tokenizer(sent))
    args.train_know_tokens, args.all_know_tokens = filtered_corpus_train[:], filtered_corpus_all[:]
    dataset_psd = []

    n_cpu = os.cpu_count() // 2
    pool = multiprocessing.Pool(n_cpu)
    if 'train' in args.mode:
        pool = multiprocessing.Pool(n_cpu)
        # dataset_psd=make(args,'train', train_dialogs, 0, len(train_dialogs))
        results = pool.starmap(make, [(args, 'train', train_dialogs, st, ed) for st, ed in [[i * len(train_dialogs) // n_cpu, (i + 1) * len(train_dialogs) // n_cpu] for i in range(n_cpu)]])
        pool.close()
        pool.join()

        dataset_psd = list(chain.from_iterable(results))
        for origin, new in zip(train_dialogs, dataset_psd):
            assert origin['goal'] == new['goal']
        print('CLEAR')
        eval(dataset_psd)
        if args.save: save(dataset_psd, 'train')
        del pool

    if 'dev' in args.mode:
        pool = multiprocessing.Pool(n_cpu)
        # dataset_psd=make(args,'dev', dev_dialogs, 0, len(dev_dialogs))
        results = pool.starmap(make, [(args, 'dev', dev_dialogs, st, ed) for st, ed in [[i * len(dev_dialogs) // n_cpu, (i + 1) * len(dev_dialogs) // n_cpu] for i in range(n_cpu)]])
        pool.close()
        pool.join()

        dataset_psd = list(chain.from_iterable(results))
        for origin, new in zip(dev_dialogs, dataset_psd):
            assert origin['goal'] == new['goal']
        print('CLEAR')
        eval(dataset_psd)
        if args.save: save(dataset_psd, 'dev')
        del pool

    if 'test' in args.mode:
        pool = multiprocessing.Pool(n_cpu)
        # dataset_psd=make(args,'test', test_dialogs, 0, len(test_dialogs))
        results = pool.starmap(make, [(args, 'test', test_dialogs, st, ed) for st, ed in [[i * len(test_dialogs) // n_cpu, (i + 1) * len(test_dialogs) // n_cpu] for i in range(n_cpu)]])
        pool.close()
        pool.join()

        dataset_psd = list(chain.from_iterable(results))
        for origin, new in zip(test_dialogs, dataset_psd):
            assert origin['goal'] == new['goal']
        print('CLEAR')
        eval(dataset_psd)
        if args.save: save(dataset_psd, 'test')
        del pool