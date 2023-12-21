import os
import json
import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizer, Trainer
import transformers
import argparse
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from datetime import datetime
from pytz import timezone
import random
import utils
import data_utils
from collections import defaultdict
from copy import deepcopy
from evaluator_conv import ConvEvaluator, ConvEvaluator_ByType
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
import numpy as np

def add_ours_specific_args(parser):
    # parser.add_argument("--asdf", action='store_true', help="~할지 여부")
    parser.add_argument( "--method", type=str, default="contriever" , help=" Method " )
    
    parser.add_argument("--contriever_max_length", type=int, default=256, help=" Goal-Topic input max_length ")
    parser.add_argument("--contriever_batch_size", type=int, default=8, help=" Method ")

    return parser

def initLogging(args):
    import git ## pip install gitpython
    filename = args.log_name #f'{args.time}_{"DEBUG" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'
    filename = os.path.join(args.log_dir, filename)
    logger.remove()
    fmt = "<green>{time:YYMMDD_HH:mm:ss}</green> | {message}"
    if not args.debug : logger.add(filename, format=fmt, encoding='utf-8')
    logger.add(sys.stdout, format=fmt, level="INFO", colorize=True)
    logger.info(f"FILENAME: {filename}")
    try: logger.info(f"Git commit massages: {git.Repo(search_parent_directories=True).head.object.hexsha[:7]}")
    except: pass
    logger.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    return logger

#------------------------------------ Main ------------------------#

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ours_main.py")
    parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)
    args = parser.parse_args()
    args.version='ko'
    # args = utils.dir_init(args, with_check=False)
    args = utils.dir_init(args, with_check=True)
    initLogging(args)
    # Read DuRec Dataset
    logger.info("Read raw file")
    if os.path.exists(os.path.join(args.data_dir, "topic2id_new.txt")) and os.path.exists(os.path.join(args.data_dir, "goal2id_new.txt")):
        topicDic = data_utils.readDic(os.path.join(args.data_dir, "topic2id_new.txt"))
        goalDic = data_utils.readDic(os.path.join(args.data_dir, "goal2id_new.txt"))
    args.topicDic, args.goalDic = topicDic, goalDic
    args.topic_num, args.goal_num = len(topicDic['int']), len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}
    logger.info(f"Read Korea raw file")
    train_dataset_raw, train_knowledge_base = data_utils.dataset_reader_ko(args, 'train')
    test_dataset_raw, valid_knowledge_base = data_utils.dataset_reader_ko(args, 'test')
    valid_dataset_raw = []
    train_dataset_aug_pred, test_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'pkl_794', f'train_pred_aug_dataset.pkl')) , utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'pkl_794', f'test_pred_aug_dataset.pkl'))
    train_dataset_pred, test_dataset_pred = utils.read_pkl("/home/work/CRSTEST/KEMGCRS/data/ko/pred_aug/gt_train_pred_aug_dataset.pkl"), utils.read_pkl("/home/work/CRSTEST/KEMGCRS/data/ko/pred_aug/gt_test_pred_aug_dataset.pkl") #utils.read_pkl("/home/work/CRSTEST/KEMGCRS/data/ko/pred_aug/gt_train_pred_aug_dataset.pkl"), utils.read_pkl("/home/work/CRSTEST/KEMGCRS/data/ko/pred_aug/gt_test_pred_aug_dataset.pkl")
    # train_dataset_pred_aug = [data for data in train_dataset_pred_aug if data['target_knowledge'] != '' and data['goal'].lower() in goal_list]
    for idx, data in enumerate(train_dataset_aug_pred):
        data['predicted_goal'] = train_dataset_pred[idx]['predicted_goal']
        data['predicted_topic'] = train_dataset_pred[idx]['predicted_topic']
        data['predicted_topic_confidence'] = train_dataset_pred[idx]['predicted_topic_confidence']

    # test_dataset_pred_aug = [data for data in test_dataset_pred_aug if data['target_knowledge'] != '' and data['goal'].lower() in goal_list]
    for idx, data in enumerate(test_dataset_aug_pred):
        data['predicted_goal'] = test_dataset_pred[idx]['predicted_goal']
        data['predicted_topic'] = test_dataset_pred[idx]['predicted_topic']
        data['predicted_topic_confidence'] = test_dataset_pred[idx]['predicted_topic_confidence']
    if args.debug: 
        train_dataset_resp, test_dataset_resp = train_dataset_aug_pred[:32], test_dataset_aug_pred[:32]
        logger.info(f"For Debugging Dataset length: 32")
    else: train_dataset_resp, test_dataset_resp = train_dataset_aug_pred, test_dataset_aug_pred
    args.bert_name = 'skt/kobert-base-v1'
    tokenizer = KoBERTTokenizer.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    bert_model = BertModel.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))


class KnowledgeDataset(Dataset):
    """ Knowledge Passage encoding --> Context
    """
    def __init__(self, args, knowledgeDB, tokenizer):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.know_max_length = args.know_max_length
        self.knowledgeDB = knowledgeDB
        self.data_samples = []

    def __getitem__(self, item):
        data = self.knowledgeDB[item]
        tokenized_data = self.tokenizer(data,
                                        max_length=self.know_max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=True)
        tokens = torch.LongTensor(tokenized_data.input_ids)
        mask = torch.LongTensor(tokenized_data.attention_mask)
        docid = self.tokenizer.encode(f"{item}", truncation=True, padding='max_length', max_length=10)[1:-1]  # 이미 Tensor로 받음
        docid = torch.LongTensor(docid)
        return tokens, mask, docid

    def __len__(self):
        return len(self.knowledgeDB)
class DialogDataset(Dataset):

    def __init__(self, args, data_sample, knowledgeDB, train_knowledgeDB, tokenizer, task, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.train_knowledgeDB = train_knowledgeDB  # new knowledge 체크용
        self.augmented_raw_sample = data_sample
        self.know_max_length = 128
        self.mode = mode
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def negative_sampler(self, target_knowledge, candidate_knowledges):

        negative_indice = []
        if len(candidate_knowledges) < self.args.negative_num:
            for idx in range(self.args.negative_num - len(candidate_knowledges)):
                negative_indice.append(0)

        while len(negative_indice) < self.args.negative_num:
            negative_idx = random.choice(candidate_knowledges)
            if (negative_idx not in negative_indice) and (negative_idx not in target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def all_negative(self, candidate_knowledges):
        all_negative = [i for i in range(len(self.knowledgeDB))]
        for candidate in candidate_knowledges:
            all_negative.remove(candidate)
        return all_negative

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        candidate_knowledges = [self.knowledgeDB.index(passage) for passage in candidate_knowledges]
        # candidate_confidences = min_max_norm(candidate_confidences)
        candidate_confidences = self.softmax(candidate_confidences)

        target_knowledge_idx = self.knowledgeDB.index(target_knowledge)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        context_batch = defaultdict()

        predicted_topic_list = deepcopy(data['predicted_topic'][:self.args.topk_topic])
        predicted_topic_confidence_list = deepcopy(data['predicted_topic_confidence'][:self.args.topk_topic])

        predicted_goal = data['predicted_goal'][0]
        
        if self.mode == 'train':
            random.shuffle(predicted_topic_list)
            candidate_topic_entities = predicted_topic_list
            predicted_topic = '|'.join(candidate_topic_entities)
        else:
            if self.args.know_item_select=='conf':
                cum_prob = 0
                candidate_topic_entities = []
                for p_topic, conf in zip(predicted_topic_list, predicted_topic_confidence_list):
                    if cum_prob < self.args.topic_conf: # or cum_prob == 0:
                        candidate_topic_entities.append(p_topic)
                        cum_prob += conf
                        # break
            elif self.args.know_item_select=='top':
                candidate_topic_entities = predicted_topic_list
            predicted_topic = '|'.join(candidate_topic_entities)
        topic_len = len(candidate_topic_entities)

        if self.args.input_prompt == 'dialog':
            prefix = ''
        elif self.args.input_prompt == 'dialog_goal':
            prefix = '<goal>' + predicted_goal + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_topic':
            prefix = '<topic>' + predicted_topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_g-topic':
            prefix = '<topic>' + topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_goal_topic':
            prefix = '<goal>' + predicted_goal + '<topic>' + predicted_topic + self.tokenizer.sep_token

        elif self.args.input_prompt == 'dialog_topic_profile':
            prefix = '<profile>' + user_profile + '<topic>' + predicted_topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_goal_profile':
            prefix = '<profile>' + user_profile + '<goal>' + predicted_goal + self.tokenizer.sep_token
        else:
            assert Exception

        prefix_encoding = self.tokenizer.encode(prefix)[1:-1][:self.know_max_length // 4]
        input_sentence = self.tokenizer('<dialog>' + dialog, add_special_tokens=False).input_ids

        input_sentence = [self.tokenizer.cls_token_id] + prefix_encoding + input_sentence[-(self.know_max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.know_max_length - len(input_sentence))

        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        context_batch['response'] = self.tokenizer(response,
                                                   add_special_tokens=True,
                                                   max_length=self.know_max_length,
                                                   padding='max_length',
                                                   truncation=True).input_ids

        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        context_batch['topic'] = self.tokenizer(topic, truncation=True, padding='max_length', max_length=32).input_ids

        candidate_confidences_pos = candidate_confidences[:self.args.pseudo_pos_num]
        candidate_knowledges_pos = candidate_knowledges[:self.args.pseudo_pos_num]

        pseudo_negative = self.negative_sampler(candidate_knowledges_pos, candidate_knowledges)

        if self.args.know_ablation == 'target':
            if target_knowledge_idx in candidate_knowledges_pos:
                candidate_knowledges_pos.remove(target_knowledge_idx)
                candidate_knowledges_pos.insert(0, target_knowledge_idx)
            else:
                candidate_knowledges_pos.insert(0, target_knowledge_idx)
                candidate_knowledges_pos = candidate_knowledges_pos[:self.args.pseudo_pos_num]
        candidate_indice = candidate_knowledges_pos + pseudo_negative  # [candidate_positives_idx[self.args.pseudo_pos_rank]]

        candidate_knowledge_text = [self.knowledgeDB[idxs] for idxs in candidate_indice]
        candidate_knowledge = self.tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=self.know_max_length)
        candidate_knowledge_token = candidate_knowledge.input_ids
        candidate_knowledge_mask = candidate_knowledge.attention_mask
        #
        context_batch['candidate_indice'] = candidate_indice
        context_batch['candidate_knowledge_token'] = candidate_knowledge_token
        context_batch['candidate_knowledge_mask'] = candidate_knowledge_mask

        context_batch['pseudo_targets'] = candidate_knowledges_pos  # [candidate_knowledges[0]]

        context_batch['target_knowledge'] = [target_knowledge_idx]  # candidate_knowledges[:3]  # target_knowledge_idx
        context_batch['all_negative'] = candidate_knowledges + self.all_negative(candidate_knowledges)
        context_batch['bm25_top20'] = candidate_knowledges
        context_batch['new_knowledge'] = self.knowledgeDB[target_knowledge_idx] not in self.train_knowledgeDB
        context_batch['isFood'] = (goal == 'Food recommendation')
        context_batch['topic_len'] = topic_len
        context_batch['candidate_topic_entities'] = [self.args.topicDic['str'][i] for i in candidate_topic_entities] + [0] * (self.args.topk_topic-len(candidate_topic_entities))
        # context_batch['candidate_topic_entities'] = context_batch['candidate_topic_entities'] + [0] * (self.args.topk_topic-len(candidate_topic_entities))
        context_batch['indices'] = idx
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)

def make_cotmae_input(save_dir, dataset_raw):
    import random
    lines = []
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    for data in dataset_raw:
        dialog = data['dialog']
        split_point = random.randint(1, len(dialog) - 1)
        samples = []
        for t in range(4):
            anchor = tokenizer.encode(tokenizer.sep_token.join(dialog[:split_point]))[1:-1]
            nearby = tokenizer.encode(tokenizer.sep_token.join(dialog[split_point:]))[1:-1]
            samples.append({"anchor": anchor, "nearby": nearby, "random_sampled": nearby, "overlap": nearby})
        lines.append({"spans": samples})
    with open("KoReDial_COTMAE.json", 'w', encoding='utf-8') as wf:
        for line in lines:
            wf.write(json.dumps(line) + "\n")