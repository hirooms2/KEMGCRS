from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from data_model_know import KnowledgeDataset
from utils import write_pkl, save_json
import numpy as np
import pickle
import json
from loguru import logger
import evaluator_conv

def knowledge_reindexing(args, knowledge_data, retriever, stage):
    # 모든 know_index를 버트에 태움
    logger.info('...knowledge indexing...(%s)' % stage)
    retriever.eval()
    knowledgeDataLoader = DataLoader(knowledge_data, batch_size=args.batch_size)
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader, bar_format=' {l_bar} | {bar:23} {r_bar}'):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        if stage == 'retrieve':
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        elif stage == 'rerank':
            knowledge_emb = retriever.rerank_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]

        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, d]
        # knowledge_emb = torch.sum(knowledge_emb * attention_mask.unsqueeze(-1), dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-20)  # [B, d]

        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index

def aug_pred_know(args, train_dataset_raw, valid_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB, bert_model, tokenizer, iter_count=0):
    from data_utils import process_augment_sample, read_pred_json_lines, eval_pred_loads, save_pred_json_lines, read_lm_pred_json_lines
    from data_model_know import KnowledgeDataset, DialogDataset
    from models.ours.retriever import Retriever
    from json import dumps
    from transformers import AutoTokenizer, AutoModel
    # args.batch_size=400
    logger.info(f"Model_Name: {args.model_name}")
    if 'contriever' in args.knowledge_method: 
    # if args.contriever or 'contriever' in args.model_name.lower():
        from models.contriever.contriever import Contriever
        # args.contriever = 'facebook/contriever'  # facebook/contriever-msmarco || facebook/mcontriever-msmarco
        args.contriever = 'facebook/contriever-msmarco' if args.version=='2' else 'facebook/mcontriever'
        bert_model = Contriever.from_pretrained(args.contriever, cache_dir=os.path.join(args.home, "model_cache", args.contriever)).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.contriever, cache_dir=os.path.join(args.home, "model_cache", args.contriever))
    elif 'cotmae' in args.knowledge_method: 
    # elif args.cotmae or 'cotmae' in args.model_name.lower(): 
        model_name = 'caskcsg/cotmae_base_uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(args.home, "model_cache", model_name))
        bert_model = AutoModel.from_pretrained(model_name, cache_dir=os.path.join(args.home, "model_cache", model_name)).to(args.device)
    elif 'dpr' in args.knowledge_method: 
    # elif 'RB_794RG_topic2' in args.model_name or 'dpr' in args.model_name.lower(): 
        tokenizer.add_special_tokens({'additional_special_tokens': ['<dialog>', '<topic>', '<goal>', '<profile>', '<situation>']})
        bert_model.resize_token_embeddings(len(tokenizer))
        pass
    
    # train_dataset_raw, valid_dataset_raw = split_validation(train_dataset_raw, args.train_ratio)
    goal_list= ['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'Food recommendation'] if args.version=='2' else ['movie recommendation', 'qa']
    train_dataset = process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB, goal_list=goal_list)
    valid_dataset = process_augment_sample(valid_dataset_raw, tokenizer, all_knowledgeDB, goal_list=goal_list)
    test_dataset = process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB, goal_list=goal_list)  # gold-topic

    # Get predicted goal, topic
    train_dataset_pred_aug = read_pred_json_lines(train_dataset, os.path.join(args.data_dir, 'pred_aug', 'goal_topic', '794', f'en_train_3711.txt'))
    test_dataset_pred_aug = read_pred_json_lines(test_dataset, os.path.join(args.data_dir, 'pred_aug', 'goal_topic', '794', f'en_test_3711.txt'))
    # 240612 LM selection 결과 넣어주기
    if args.inspired:
        train_file_path = os.path.join(args.home, 'data/2/inspired/train_pred_aug_dataset_inspired_new2.pkl')
        train_dataset_pred_aug = pickle.load(open(file=train_file_path, mode='rb'))

        test_file_path = os.path.join(args.home, 'data/2/inspired/test_pred_aug_dataset_inspired_new3.pkl')
        test_dataset_pred_aug = pickle.load(open(file=test_file_path, mode='rb'))
    
    if args.LM_selection:
        test_dataset_pred_aug = read_lm_pred_json_lines(test_dataset, os.path.join(args.data_dir, 'pred_aug', 'goal_topic', '794', f'en_test_lm_3711_321.json'))

    eval_pred_loads(test_dataset_pred_aug, task='topic')

    # Get Pseudo label
    # inspired에서 pseudo label 넣는 부분 빼주기 위해서 임시작업
    if not args.inspired:
        # 원래는 다 살아있었음
        logger.info(f"Get Pseudo Label {args.pseudo_labeler.upper()}")
        train_dataset_pred_aug = read_pred_json_lines(train_dataset_pred_aug, os.path.join(args.data_dir, 'pseudo_label', args.pseudo_labeler, f'en_train_pseudo_BySamples3711.txt'))
        test_dataset_pred_aug = read_pred_json_lines(test_dataset_pred_aug, os.path.join(args.data_dir, 'pseudo_label', args.pseudo_labeler, f'en_test_pseudo_BySamples3711.txt'))
    # 240612 LM selection 결과 넣어주기
    
    eval_pred_loads(test_dataset_pred_aug, task='label')

    if args.debug: train_dataset_pred_aug, test_dataset_pred_aug = train_dataset_pred_aug[:30], test_dataset_pred_aug[:30]

    train_datamodel_know = DialogDataset(args, train_dataset_pred_aug, train_knowledgeDB, train_knowledgeDB, tokenizer, mode='train', task='know')
    # valid_datamodel_know = DialogDataset(args, valid_dataset_pred_aug, all_knowledgeDB, train_knowledgeDB, tokenizer, mode='test', task='know')
    test_datamodel_know = DialogDataset(args, test_dataset_pred_aug, all_knowledgeDB, train_knowledgeDB, tokenizer, mode='test', task='know')

    train_dataloader_retrieve = DataLoader(train_datamodel_know, batch_size=args.batch_size*10, shuffle=False)
    test_dataloader_retrieve = DataLoader(test_datamodel_know, batch_size=args.batch_size*10, shuffle=False)


    retriever = Retriever(args, bert_model)
    logger.info(f"Knowledge Method: {args.knowledge_method}, Model name: {args.model_name} , Topic_Conf: {args.topic_conf} Topic_Top-K: {args.topk_topic}")
    retriever.load_state_dict(torch.load(os.path.join(args.saved_model_path,'save','our' if args.idea else 'default', args.knowledge_method,f"{args.model_name}.pt"), map_location=args.device), strict=False)
    # retriever.load_state_dict(torch.load(f"{args.saved_model_path}/{args.model_name}_know.pt", map_location='cuda:0'), strict=False)
    logger.info(f"Loaded model: {args.saved_model_path}/{args.knowledge_method}/{args.model_name}.pt")

    if args.sort_candidates:
        model_name = args.model_name + "hit_100_sort"
    else:
        model_name = args.model_name

    with torch.no_grad():
        retriever.to(args.device)
       
        hitdic_ratio, train_output_str, train_top10_cand_knows, train_top10_cand_knows_conf = eval_know(args, train_dataloader_retrieve, retriever, train_knowledgeDB, tokenizer, data_type='train')
        save_pred_know_json(os.path.join(args.output_dir, f"en_{model_name}_{iter_count}_train_know_3711.txt"), train_top10_cand_knows, train_top10_cand_knows_conf)
       
        hitdic_ratio, test_output_str, test_top10_cand_knows, test_top10_cand_knows_conf = eval_know(args, test_dataloader_retrieve, retriever, all_knowledgeDB, tokenizer, data_type='test')
        save_pred_know_json(os.path.join(args.output_dir, f"en_{model_name}_{iter_count}_test_know_3711.txt"), test_top10_cand_knows, test_top10_cand_knows_conf)
    for i in train_output_str:
        logger.info(f"{model_name}: {i}")
    for i in test_output_str:
        logger.info(f"{model_name}: {i}")
    torch.cuda.empty_cache()


def save_pred_know_json(data_path, top10_cand_knows, top10_cand_knows_conf):
    from json import dumps
    if os.path.exists(data_path): data_path = data_path[:-4]+"_copy.txt"
    logger.info(f"New output: {data_path}")
    with open(data_path, 'a', encoding='utf8') as fw:
        for k,c in zip(top10_cand_knows, top10_cand_knows_conf):
            fw.write(dumps({'predicted_know' : k[:10],'predicted_know_confidence' : c[:10]}) + "\n") # 10개 넣어주기
            # fw.write(dumps({'predicted_know' : k[:5],'predicted_know_confidence' : c[:5]}) + "\n") # 5개 넣어주기


def eval_know(args, test_dataloader, retriever, knowledgeDB, tokenizer, write=None, retrieve=None, data_type='test'):
    logger.info(args.stage)
    retriever.eval()
    # Read knowledge DB
    # knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    # knowledge_index = knowledge_index.to(args.device)
    jsonlineSave = []
    # bert_model = bert_model.to(args.device)
    new_cnt = 0
    logger.info('Knowledge indexing for test')
    knowledge_data = KnowledgeDataset(args, knowledgeDB, tokenizer)  # knowledge dataset class

    knowledge_index_rerank = knowledge_reindexing(args, knowledge_data, retriever, stage='rerank')
    knowledge_index_rerank = knowledge_index_rerank.to(args.device)

    goal_list = ['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'Food recommendation', 'Chat about stars']
    hit1_goal, hit3_goal, hit5_goal, hit10_goal, hit20_goal = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    hit1, hit5, hit3, hit10, hit20 = [], [], [], [], []
    hit1_new, hit5_new, hit3_new, hit10_new, hit20_new = [], [], [], [], []
    hit1_topic = []
    hit20_p1, hit20_p2, hit20_p3, hit20_p23 = [], [], [], []

    cnt = 0

    pred = []
    targets = []
    current = 0
    topic_lens = []
    contexts, responses, g_goals, g_topics, is_new_knows = [],[],[],[],[]
    top10_cand_knows, top10_cand_knows_conf, target_knows=[],[],[]
    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):  # TODO: Knowledge task 분리중
        batch_size = batch['attention_mask'].size(0)
        dialog_token = batch['input_ids']
        dialog_mask = batch['attention_mask']
        response = batch['response']
        new_knowledge = batch['new_knowledge']
        # candidate_topic_entities = batch['candidate_topic_entities']

        topic_lens.extend(batch['topic_len'].tolist())
        # candidate_indice = batch['candidate_indice']
        # candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
        # candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]

        batch_goals = [args.goalDic['int'][int(idx)] for idx in batch['goal_idx']]
        batch_topics = [args.topicDic['int'][int(idx)] for idx in batch['topic_idx']]

        # candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
        target_knowledge_idx = batch['target_knowledge']

        # if args.stage == 'retrieve':
        dot_score = retriever.compute_know_score_candidate(dialog_token, dialog_mask, knowledge_index_rerank)  # todo: DPR용 (1/2)

        if write:
            for batch_id in range(batch_size):
                top_candidate = torch.topk(dot_score[batch_id], k=5, dim=0).indices  # [B, K]
                input_text = tokenizer.decode(dialog_token[batch_id], skip_special_tokens=True)
                target_knowledge_text = knowledgeDB[int(target_knowledge_idx[batch_id])] #for i in target_knowledge_idx[batch_id] # knowledgeDB[target_knowledge_idx]
                retrieved_knowledge_text = [knowledgeDB[idx].lower() for idx in top_candidate]  # list
                correct = target_knowledge_idx[batch_id] in top_candidate
                ground_topic = args.topicDic['int'][batch['topic_idx'][batch_id].item()]
                # candidate_topic = [args.topicDic['int'][i.item()] for i in candidate_topic_entities[batch_id][:topic_lens[batch_id]]]
                # selected_topic = -1
                # for i, topic in enumerate(candidate_topic):
                #     if topic in retrieved_knowledge_text[0]:
                #         selected_topic = i
                #         break
                rec_hit = ground_topic in retrieved_knowledge_text[0]

                gen_response = tokenizer.decode(response[batch_id], skip_special_tokens=True)
                # jsonlineSave.append(
                #     {'goal_type': args.goalDic['int'][batch['goal_idx'][batch_id].item()], 'topic': ground_topic, 'passage_hit': correct, 'dialog': input_text, 'target': target_knowledge_text, 'response': gen_response, "predict5": retrieved_knowledge_text, 'topic_len': batch['topic_len'].tolist()[0],
                #      'candidate_topic_entities': candidate_topic, 'selected_topic':selected_topic,'rec_hit': rec_hit})
                jsonlineSave.append(
                    {'goal_type': args.goalDic['int'][batch['goal_idx'][batch_id].item()], 'topic': ground_topic, 'passage_hit': correct, 'dialog': input_text, 'target': target_knowledge_text, 'response': gen_response, "predict5": retrieved_knowledge_text, 'topic_len': batch['topic_len'].tolist()[0],
                     'rec_hit': rec_hit})
            # save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)
        top10_know_tmp = [[knowledgeDB[int(idx)] for idx in top10] for top10 in torch.topk(dot_score, k=10).indices]
        top10_conf_tmp = [[float(j) for j in i] for i in torch.topk(dot_score, k=10).values]
        target_know_tmp = [knowledgeDB[int(top10)] for top10 in target_knowledge_idx]
        top10_know_sort, top10_conf_sort = [], []
        if args.sort_candidates:
            for candidates, confs, target in zip(top10_know_tmp, top10_conf_tmp, target_know_tmp):
                if target in candidates:
                    idx = candidates.index(target)
                    if idx != 0:
                        conf = confs[0]
                        candidates.pop(idx)
                        confs.pop(idx)
                        candidates.insert(0,target)
                        confs.insert(0,conf)
                else:
                    conf = confs[0]
                    candidates.pop()
                    confs.pop()
                    candidates.insert(0,target)
                    confs.insert(0,conf)
                
                top10_know_sort.append(candidates)
                top10_conf_sort.append(confs)
            top10_cand_knows.extend(top10_know_sort)
            top10_cand_knows_conf.extend(top10_conf_sort)
        else:    
            top10_cand_knows.extend(top10_know_tmp) # 이건 왜 =임? extend 해야 하는거아님?
            top10_cand_knows_conf.extend(top10_conf_tmp)
        contexts.extend(tokenizer.batch_decode(dialog_token, skip_special_tokens=False))
        responses.extend(tokenizer.batch_decode(response, skip_special_tokens=False))
        is_new_knows.extend([idx.item() for idx in new_knowledge])
        g_goals.extend(batch_goals)
        g_topics.extend(batch_topics)
        target_knows.extend(target_know_tmp)

    hitdic, hitdic_ratio, output_str = evaluator_conv.know_hit_ratio(args, pred_pt=top10_cand_knows, gold_pt=target_knows, new_knows=is_new_knows, types=g_goals)
    topic_len_avg = np.average(topic_lens)


    if retrieve:
        with open(f'augmented_dataset_{data_type}.txt', 'wb') as f:
            pickle.dump(test_dataloader.dataset.augmented_raw_sample, f)

    if write:
        # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
        # filename = f"{args.output_dir}/eval_know_json.pkl"
        write_pkl(obj=jsonlineSave, filename= os.path.join(args.output_dir, 'best_model_best_setting.pkl'))  # 입출력 저장
        # save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)

    logger.info(f"avg topic: %.2f" % topic_len_avg)

    return hitdic_ratio, output_str, top10_cand_knows, top10_cand_knows_conf

