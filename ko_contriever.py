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

import utils
import data_utils
from collections import defaultdict
from copy import deepcopy
from evaluator_conv import ConvEvaluator, ConvEvaluator_ByType


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
    