import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from transformers import AutoConfig, AutoTokenizer, AutoModel

from data_model_know import KnowledgeDataset, DialogDataset
from data_utils import process_augment_sample, read_pred_json_lines,eval_pred_loads, save_pred_json_lines
from model_play.ours.eval_know_retrieve import knowledge_reindexing, eval_know  #### Check
# from models.ours.cotmae import BertForCotMAE
from utils import *
# from models import *
import logging
import numpy as np
from loguru import logger
import os


def update_key_bert(key_bert, query_bert):
    logger.info('update moving average')
    decay = 0  # If 0 then change whole parameter
    for current_params, ma_params in zip(query_bert.parameters(), key_bert.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = decay * old_weight + (1 - decay) * up_weight


def train_know(args, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB, bert_model, tokenizer):
    from models.ours.retriever import Retriever
    retriever = Retriever(args, bert_model)
    # retriever.load_state_dict(torch.load("model_save/2/RB_794RG_topic2_conf80_hj_know.pt", map_location='cuda:0')); logger.info('LOAD BEST MODEL TH#####################################################')
    args.know_topk = 5

    retriever = retriever.to(args.device)
    goal_list = []
    if 'Movie' in args.goal_list: goal_list.append('Movie recommendation')
    if 'POI' in args.goal_list: goal_list.append('POI recommendation')
    if 'Music' in args.goal_list: goal_list.append('Music recommendation')
    if 'QA' in args.goal_list: goal_list.append('Q&A'); goal_list.append('QA')
    if 'Food' in args.goal_list: goal_list.append('Food recommendation')
    goal_list = [goal.lower() for goal in goal_list]
    # if 'Chat' in args.goal_list:  goal_list.append('Chat about stars')
    logger.info(f" Goal List in Knowledge Task : {args.goal_list}")
    # if args.version =='ko':
    #     train_dataset_pkl = read_pkl("/home/work/CRSTEST/KEMGCRS/data/ko/pred_aug/gt_train_pred_aug_dataset.pkl")
    #     test_dataset_pkl = read_pkl("/home/work/CRSTEST/KEMGCRS/data/ko/pred_aug/gt_test_pred_aug_dataset.pkl")
    #     eval_pred_loads(test_dataset_pred_aug, task='topic')
    #     eval_pred_loads(test_dataset_pred_aug, task='label')
    # else: 
        # train_dataset_raw, valid_dataset_raw = split_validation(train_dataset_raw, args.train_ratio)
    train_dataset = process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB, goal_list=goal_list)
    if valid_dataset_raw: valid_dataset = process_augment_sample(valid_dataset_raw, tokenizer, all_knowledgeDB, goal_list=goal_list)
    test_dataset = process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB, goal_list=goal_list)  # gold-topic

    # train_dataset_pkl = read_pkl(os.path.join(args.data_dir, 'pred_aug', 'pkl_794', f'train_pred_aug_dataset.pkl')) # Topic 0.793
    # test_dataset_pkl = read_pkl(os.path.join(args.data_dir, 'pred_aug', 'pkl_794', f'test_pred_aug_dataset.pkl'))
    # save_pred_json_lines(train_dataset_pred_aug , os.path.join(args.data_dir, 'pred_aug', 'goal_topic', '794', f'en_train_3711.txt') , ['predicted_goal', 'predicted_goal_confidence', 'predicted_topic','predicted_topic_confidence']) # list(filter(lambda x: len(x['candidate_knowledges'])>0 and x['goal'] in ['Movie Recommendation','QA'], test_dataset_pkl))
    # Get predicted goal, topic
    train_dataset_pred_aug = read_pred_json_lines(train_dataset, os.path.join(args.data_dir, 'pred_aug', 'goal_topic', '794', f'en_train_3711.txt'))
    test_dataset_pred_aug = read_pred_json_lines(test_dataset, os.path.join(args.data_dir, 'pred_aug', 'goal_topic', '794', f'en_test_3711.txt'))
    eval_pred_loads(test_dataset_pred_aug, task='topic')

    # Get Pseudo label
    logger.info(f" Get Pseudo Label {args.pseudo_labeler.upper()}")
    train_dataset_pred_aug = read_pred_json_lines(train_dataset_pred_aug, os.path.join(args.data_dir, 'pseudo_label', args.pseudo_labeler, f'en_train_pseudo_BySamples3711.txt'))
    test_dataset_pred_aug = read_pred_json_lines(test_dataset_pred_aug, os.path.join(args.data_dir, 'pseudo_label', args.pseudo_labeler, f'en_test_pseudo_BySamples3711.txt'))
    eval_pred_loads(test_dataset_pred_aug, task='label')



    # train_dataset_pred_aug = [data for data in train_dataset_pred_aug if data['target_knowledge'] != '' and data['goal'].lower() in goal_list]
    # for idx, data in enumerate(train_dataset):
    #     data['predicted_goal'] = train_dataset_pred_aug[idx]['predicted_goal']
    #     data['predicted_topic'] = train_dataset_pred_aug[idx]['predicted_topic']
    #     data['predicted_topic_confidence'] = train_dataset_pred_aug[idx]['predicted_topic_confidence']

    # test_dataset_pred_aug = [data for data in test_dataset_pred_aug if data['target_knowledge'] != '' and data['goal'].lower() in goal_list]
    # for idx, data in enumerate(test_dataset):
    #     data['predicted_goal'] = test_dataset_pred_aug[idx]['predicted_goal']
    #     data['predicted_topic'] = test_dataset_pred_aug[idx]['predicted_topic']
    #     data['predicted_topic_confidence'] = test_dataset_pred_aug[idx]['predicted_topic_confidence']

    if args.debug: train_dataset_pred_aug, test_dataset_pred_aug = train_dataset_pred_aug[:30], test_dataset_pred_aug[:30]

    train_datamodel_know = DialogDataset(args, train_dataset_pred_aug, train_knowledgeDB, train_knowledgeDB, tokenizer, mode='train', task='know')
    # valid_datamodel_know = DialogDataset(args, valid_dataset_pred_aug, all_knowledgeDB, train_knowledgeDB, tokenizer, mode='test', task='know')
    test_datamodel_know = DialogDataset(args, test_dataset_pred_aug, all_knowledgeDB, train_knowledgeDB, tokenizer, mode='test', task='know')

    train_dataloader = DataLoader(train_datamodel_know, batch_size=args.batch_size, shuffle=True)
    train_dataloader_retrieve = DataLoader(train_datamodel_know, batch_size=args.batch_size, shuffle=False)
    # valid_dataloader = DataLoader(test_datamodel_know, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_datamodel_know, batch_size=args.batch_size, shuffle=False)
    test_dataloader_write = DataLoader(test_datamodel_know, batch_size=1, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    # eval_know(args, test_dataloader, retriever, all_knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리
    # eval_know(args, test_dataloader_write, retriever, all_knowledgeDB, tokenizer, write=True)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

    # eval_metric, best_output, best_epoch = [-1], None, 0
    eval_metric, best_output, best_epoch, best_hitdic = [-1], None, 0, None
    result_path = f"{args.time}_{args.model_name}_result"

    for epoch in range(args.num_epochs):
        train_epoch_loss = 0
        num_update = 0
        retriever.train()
        for batch in tqdm(train_dataloader, desc="Knowledge_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            dialog_token = batch['input_ids']
            dialog_mask = batch['attention_mask']
            goal_idx = batch['goal_idx']
            # response = batch['response']
            candidate_indice = batch['candidate_indice']
            candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,2,256]
            candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,2,256]
            # sampling_results = batch['sampling_results']

            target_knowledge_idx = batch['target_knowledge']  # [B,5,256]

            logit_pos, logit_neg = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_indice, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
            if args.train_ablation_reverse: # Relevance-degree 에 기반한 RGL의 효과 검증을 위해, relevance-degree의 역순으로 subgroup 을 만들었을 때와 비교
                logit_pos = torch.flip(logit_pos, dims=(1,))

            cumsum_logit = torch.cumsum(logit_pos, dim=1)  # [B, K]  # Grouping

            loss = 0
            # pseudo_confidences_mask = batch['pseudo_confidences']  # [B, K]
            for idx in range(args.pseudo_pos_num):
                # confidence = torch.softmax(pseudo_confidences[:, :idx + 1], dim=-1)
                   # g_logit = torch.sum(logit_pos[:, :idx + 1] * pseudo_confidences_mask[:, :idx + 1], dim=-1) / (torch.sum(pseudo_confidences_mask[:, :idx + 1], dim=-1) + 1e-20)

                if args.train_ablation == 'CL':# Contrastive loss --> 이게 그냥인것
                    g_logit = logit_pos[:, idx]  # For Sampling
                if args.train_ablation == 'RG' or args.train_ablation == 'GL':# GCL
                    g_logit = cumsum_logit[:, idx] / (idx + 1)  # For GCL!!!!!!! (our best)

                g_logit = torch.cat([g_logit.unsqueeze(1), logit_neg], dim=1)
                g_loss = (-torch.log_softmax(g_logit, dim=1).select(dim=1, index=0)).mean()
                loss += g_loss
            if args.train_ablation == 'GL':
                loss = g_loss

            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_update += 1

        scheduler.step()

        logger.info(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")

        hitdic_ratio, output_str, _, _ = eval_know(args, test_dataloader, retriever, all_knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리
        # hit1, hit3, hit5, hit10, hit20, hit_movie_result, hit_music_result, hit_qa_result, hit_poi_result, hit_food_result, hit_chat_result, hit1_new, hit3_new, hit5_new, hit10_new, hit20_new = eval_know(args, test_dataloader, retriever, all_knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리
        for i in output_str:
            logger.info(f"EPOCH_{epoch}: {i}")

        if hitdic_ratio['total']['hit1'] >= eval_metric[0]:
            best_output = output_str
            best_epoch = epoch
            best_hitdic = hitdic_ratio
            eval_metric[0] = hitdic_ratio['total']['hit1']
            # torch.save(retriever.state_dict(), os.path.join(args.saved_model_path, f"{args.model_name}_know.pt"))  # TIME_MODELNAME 형식
            torch.save(retriever.state_dict(), os.path.join(args.saved_model_path, f"{args.model_name}_know_top_{args.topk_topic}.pt"))  # TIME_MODELNAME 형식 03/11 JP 실험

    hitdic_ratio, output_str, _, _ = eval_know(args, test_dataloader, retriever, all_knowledgeDB, tokenizer, write=True)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리
    return best_hitdic, best_output
