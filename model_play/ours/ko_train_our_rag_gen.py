import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import optim
from loguru import logger
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
# from transformers import DPRContextEncoder,DPRContextEncoderTokenizerFast, RagRetriever
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagSequenceForGeneration, RagTokenizer, BartForConditionalGeneration
from typing import List
from datasets import Features, Sequence, Value, load_dataset
from functools import partial
import faiss
from data_model_know import KnowledgeDataset
from model_play.ours.eval_know_retrieve import knowledge_reindexing
from evaluator_conv import ConvEvaluator, ConvEvaluator_ByType
from models.ours.retriever import Retriever
import utils
import data_model
from torcheval.metrics.functional.text import perplexity
from copy import deepcopy
from data_utils import process_augment_sample, read_pred_json_lines,eval_pred_loads, save_pred_json_lines

def make_aug_gt_pred(args, bert_model, tokenizer, train_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB):
    import data_utils
    import data_model
    from models.ours.retriever import Retriever
    from model_play.ours.train_bert_goal_topic import eval_goal_topic_model

    if args.rag_train_alltype:
        train_dataset = data_utils.process_augment_all_sample(train_dataset_raw, tokenizer, train_knowledgeDB)  ## 42086
    else:
        train_dataset = data_utils.process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB)  ##11621
    if args.rag_test_alltype:
        test_dataset = data_utils.process_augment_all_sample(test_dataset_raw, tokenizer, all_knowledgeDB)  ## 13282
    else:
        test_dataset = data_utils.process_augment_sample(test_dataset_raw, tokenizer, all_knowledgeDB)  ## 3711

    logger.info(f"Train AllType: {args.rag_train_alltype}: {len(train_dataset)}, Test AllType: {args.rag_test_alltype}: {len(test_dataset)}")

    retriever = Retriever(args, bert_model)  # eval_goal_topic_model 함수에서 goal, topic load해서 쓸것임
    retriever.to(args.device)
    train_datamodel_topic = data_model.GenerationDataset(args, train_dataset, train_knowledgeDB, tokenizer, mode='train', subtask=args.subtask)
    test_datamodel_topic = data_model.GenerationDataset(args, test_dataset, all_knowledgeDB, tokenizer, mode='test', subtask=args.subtask)

    train_GT_pred_auged_Dataset, test_GT_pred_auged_Dataset = eval_goal_topic_model(args, train_datamodel_topic, test_datamodel_topic, retriever, tokenizer)
    train_gt_pred_auged, test_gt_pred_auged = train_GT_pred_auged_Dataset.augmented_raw_sample, test_GT_pred_auged_Dataset.augmented_raw_sample
    return train_gt_pred_auged, test_gt_pred_auged

def make_know_pred(args, our_best_model, tokenizer, aug_Dataset, knowledge_index_rerank,  all_knowledgeDB):
    dataloader = DataLoader(aug_Dataset, batch_size=args.rag_batch_size*4, shuffle=False)

    types, pred_know_texts, pred_know_confs, label_gold_knowledges = [], [], [],[]
    for batch in tqdm(dataloader, desc=f"Epoch {'KnowRetrieve'}__{0}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["response"].to(args.device)
        q_vector = our_best_model.query_bert(source_ids, source_mask).last_hidden_state[:, 0, :]
        doc_all_scores = (q_vector @ knowledge_index_rerank.transpose(1, 0))
        retrieved_doc_ids = torch.topk(doc_all_scores, k=5).indices
        
        # types.extend([args.goalDic['int'][int(idx)] for idx in batch['goal_idx']])
        pred_know_texts.extend([[dataloader.dataset.knowledgeDB[int(j)] for j in i ] for i in retrieved_doc_ids])
        pred_know_confs.extend([[float(j) for j in i] for i in torch.topk(doc_all_scores, k=5).values])
        types.extend([args.taskDic['goal']['int'][int(i)] for i in batch['goal_idx']])
        label_gold_knowledges.extend(batch['target_knowledge_label'])
    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=pred_know_texts, gold_pt=label_gold_knowledges, new_knows=None, types=types)
    for i in output_str:
        logger.info(f"Knowledge_Check: {i}")
    for i, dataset in enumerate(aug_Dataset.augmented_raw_sample):
        dataset[f"predicted_know"] = pred_know_texts[i]
        dataset[f"predicted_know_confidence"] = pred_know_confs[i]

    return aug_Dataset.augmented_raw_sample



def train_KO_our_rag_generation(args, bert_model, tokenizer, train_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB):
    # logger.info("Retrieve된 input으로 받아서 생성 (context_input_ids)")
    from models.kobart import get_pytorch_kobart_model, get_kobart_tokenizer
    from model_play.rag import rag_retrieve
    assert args.version=='ko', "Version Must Set to 'ko' !!! "
    logger.info(f"\n  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA\nOUR KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA\n  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA  KOREA\n")
    logger.info(f"\n\nOUR {args.rag_our_model}_BERT_Retriever model For resp, RAG_OUR_MODEL: {args.rag_our_model}, RAG_OnlyDecoderTune: {args.rag_onlyDecoderTune}\n\n")
    # if args.rag_onlyDecoderTune: args.rag_batch_size = args.rag_batch_size*2
    ## Topic pre-trained bert를 rag의 시작점으로 잡아보면 어떨까? 
    # train_dataset_aug_pred, test_dataset_aug_pred = make_aug_gt_pred(args, deepcopy(bert_model), tokenizer, train_dataset_raw, test_dataset_raw, train_knowledgeDB, all_knowledgeDB)
    goal_list = ['movie recommendation', 'qa']
    train_dataset = process_augment_sample(train_dataset_raw, tokenizer, train_knowledgeDB, goal_list=goal_list)
    # if valid_dataset_raw: valid_dataset = process_augment_sample(valid_dataset_raw, tokenizer, all_knowledgeDB, goal_list=goal_list)
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




    train_dataset_aug_pred,test_dataset_aug_pred = train_dataset_pred_aug[:], test_dataset_pred_aug[:]
    logger.info(f"Length of Pred_Auged Train,Test: {len(train_dataset_aug_pred)}, {len(test_dataset_aug_pred)}")
    if args.debug: train_dataset_aug_pred, test_dataset_aug_pred = train_dataset_aug_pred[:10], test_dataset_aug_pred[:10]
    # tokenizer.add_special_tokens({'additional_special_tokens': ['<dialog>', '<topic>', '<goal>', '<profile>', '<situation>'],})
    # bert_model.resize_token_embeddings(len(tokenizer))
    our_best_model = Retriever(args, deepcopy(bert_model))
    if args.rag_our_model.upper() == 'C2DPR': # 
        load_model_name = os.path.join(args.saved_model_path, f"GCL2_topic3_conf60_KO_retriever.pt")
        tokenizer.add_special_tokens({'additional_special_tokens': ['<dialog>', '<topic>', '<goal>', '<profile>', '<situation>'],})
        bert_model.resize_token_embeddings(len(tokenizer))
        our_best_model = Retriever(args, deepcopy(bert_model))
        logger.info(f"@@@@@Load Our C2DPR RAG On Bert : {load_model_name}")
        our_best_model.load_state_dict(torch.load(load_model_name, map_location=args.device), strict=False)
        our_best_model.query_bert.name_or_path, our_best_model.rerank_bert.name_or_path = "skt/c2dpr_query_bert", "skt/c2dpr_rerank_bert"
        # our_best_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"DPR_retriever.pt"), map_location=args.device), strict=False) # C2DPR이 아직 없어서 temp
    elif args.rag_our_model.upper() == 'DPR': 
        load_model_name = os.path.join(args.saved_model_path, f"DPR_retriever.pt")
        logger.info(f"@@@@@Load Our DPR RAG On Bert: {load_model_name}")
        our_best_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"DPR_retriever.pt"), map_location=args.device), strict=False)
        our_best_model.query_bert.name_or_path, our_best_model.rerank_bert.name_or_path = "skt/dpr_query_bert", "skt/dpr_rerank_bert"
    elif "contriever" in args.rag_our_model.lower():
        from models.contriever.contriever import Contriever
        from transformers import AutoTokenizer
        args.contriever = "facebook/mcontriever" #, "facebook/mcontriever-msmarco"
        bert_model = Contriever.from_pretrained(args.contriever, cache_dir=os.path.join(args.home, "model_cache", args.contriever)).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.contriever, cache_dir=os.path.join(args.home, "model_cache", args.contriever))
        our_best_model = Retriever(args, deepcopy(bert_model))
        if '1' in args.idea and '2' in args.idea: 
            our_best_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"contriever_our.pt"), map_location=args.device), strict=False)
            our_best_model.query_bert.name_or_path, our_best_model.rerank_bert.name_or_path = "facebook/mcontriever-query", "facebook/mcontriever-rerank"
        else: 
            our_best_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"contriever_noidea.pt"), map_location=args.device), strict=False)
    else: logger.info("@@@@@Load Default RAG On Bert")
    
    our_best_model.to(args.device)
    our_question_encoder = deepcopy(our_best_model.query_bert)
    our_ctx_encoder = deepcopy(our_best_model.rerank_bert)
    logger.info(f"OUR_question_encoder: {our_question_encoder.encoder.layer[0].attention.self.key.weight[0][:50][0]}")
    logger.info(f"     OUR_ctx_encoder: {our_ctx_encoder.encoder.layer[0].attention.self.key.weight[0][:50][0]}")

    knowledgeDB_list = list(all_knowledgeDB)
    knowledgeDB_csv_path = os.path.join(args.data_dir, 'rag')  # HOME/data/2/rag/"train_knowledge.csv")
    utils.checkPath(knowledgeDB_csv_path)
    knowledgeDB_csv_path = os.path.join(knowledgeDB_csv_path, f'my_knowledge_dataset_{args.gpu}' + ('_debug.csv' if args.debug else '.csv'))
    args.knowledgeDB_csv_path = knowledgeDB_csv_path
    with open(knowledgeDB_csv_path, 'w', encoding='utf-8') as f:
        for know in knowledgeDB_list:
            tmp=know.replace('\t',' ')
            f.write(f" \t{tmp}\n")
    #
    faiss_dataset = load_dataset("csv", data_files=[knowledgeDB_csv_path], split="train", delimiter="\t", column_names=["title", "text"])
    faiss_dataset = faiss_dataset.map(rag_retrieve.split_documents, batched=True, num_proc=1)

    MODEL_CACHE_DIR = os.path.join(args.home, 'model_cache', 'facebook/dpr-ctx_encoder-multiset-base')

    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR).to(device=args.device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR)

    
    # logger.info("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@ Use ko-Bert For ctx_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    ctx_encoder.ctx_encoder.bert_model = deepcopy(bert_model) # SKT-bert
    ctx_tokenizer = tokenizer # SKT-bert tokenizer
    if args.rag_our_model: # 학습된 KO리트리버의 our_best_model.query_bert
        # logger.info("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@ Use Our Trained Bert For ctx_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Use Our Trained Bert For ctx_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
        ctx_encoder.ctx_encoder.bert_model = our_ctx_encoder
        ctx_tokenizer = tokenizer

    logger.info(f"ctx_encoder의 bert.encoder.attn.key.weight: {ctx_encoder.ctx_encoder.bert_model.encoder.layer[0].attention.self.key.weight[0][:50][0]}")
    logger.info("Create Knowledge Dataset")
    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})  # optional, save as float32 instead of float64 to save space
    ctx_encoder.eval()
    ctx_encoder.to(args.device)
    faiss_dataset = faiss_dataset.map(
        partial(rag_retrieve.embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, args=args),
        batched=True, batch_size=args.rag_batch_size, features=new_features, )

    passages_path = os.path.join(args.data_dir, 'rag', f"my_knowledge_dataset_{args.gpu}")
    if args.debug: passages_path += '_debug'
    args.passages_path = passages_path
    faiss_dataset.save_to_disk(passages_path)

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    faiss_dataset.add_faiss_index('embeddings', custom_index=index)
    # faiss_dataset.add_faiss_index(column='embeddings', index_name = 'embeddings', custom_index=index, faiss_verbose=True)
    print(f"Length of Knowledge knowledge_DB : {len(faiss_dataset)}")

    kobart_tokenizer = get_kobart_tokenizer(cachedir=os.path.join(args.home,'model_cache','kobart'))
    kobart_tokenizer.name_or_path = 'skt/kobart_tokenizer'
    # kobart_tokenizer.add_special_tokens({'additional_special_tokens': ['<dialog>', '<topic>', '<type>', '<user_profile>', '<situation>','user: ','system: '],})
    kobart = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model(cachedir=os.path.join(args.home,'model_cache','kobart'))).to(args.device)
    kobart.resize_token_embeddings(len(kobart_tokenizer))
    kobart.name_or_path='skt/kobart'
    ### MODEL CALL
    retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom', indexed_dataset=faiss_dataset, init_retrieval=True)
    retriever.set_ctx_encoder_tokenizer(ctx_tokenizer)  # NO TOUCH
    retriever.generator_tokenizer = kobart_tokenizer
    retriever.question_encoder_tokenizer = tokenizer 

    rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever).to(args.device)
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    # if args.rag_ctx_training: 
    # rag_model.set_context_encoder_for_training(ctx_encoder) # All Fine-tune 때 쓰던 코드같은데 이거 키면 ctx_encoder가 학습됨
    
    logger.info("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@ Model Ko-BERT to rag.question_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    rag_model.rag.question_encoder.question_encoder.bert_model = our_question_encoder
    # rag_model.rag.question_encoder.question_encoder.bert_model = deepcopy(bert_model)
    rag_tokenizer.question_encoder = tokenizer


    rag_model.generator = kobart
    rag_model.rag.generator = kobart
    rag_tokenizer.generator = kobart_tokenizer
    # rag_model.rag.ctx_encoder.ctx_encoder.bert_model = deepcopy(ctx_encoder.ctx_encoder.bert_model)


    if args.rag_our_model:
        logger.info("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@ Model question_encoder changed by ours @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@ Model question_encoder changed by ours @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
        rag_model.rag.question_encoder.question_encoder.bert_model = our_question_encoder
        rag_tokenizer.question_encoder = tokenizer
    logger.info(f"RAG Model question_encoder layer0.key.weight: {rag_model.rag.question_encoder.question_encoder.bert_model.encoder.layer[0].attention.self.key.weight[0][:50][0]}")
    
    # if args.rag_our_model:
    train_Dataset = data_model.RagDataset(args, train_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='train')
    test_Dataset = data_model.RagDataset(args, test_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='test')
    # else:
    #     ## UniMIND Dataset으로 Sch rag에 넣기
    #     args.uni_max_input_length, args.uni_max_target_length = args.rag_max_input_length, args.rag_max_target_length
    #     train_Dataset = data_model.UnimindDataset(args, train_dataset_aug_pred, rag_tokenizer, mode='train', method='unimind')
    #     test_Dataset = data_model.UnimindDataset(args, test_dataset_aug_pred, rag_tokenizer, mode='test', method='unimind')
    
    if args.rag_our_model or args.rag_our_model:
        knowledge_data = KnowledgeDataset(args, all_knowledgeDB, tokenizer)  # KTH: FOR RETRIEVE
        knowledge_index_rerank = knowledge_reindexing(args, knowledge_data, our_best_model, stage='rerank')  # KTH: FOR RETRIEVE
        knowledge_index_rerank = knowledge_index_rerank.to(args.device)  # KTH: FOR RETRIEVE
        know_aug_train_dataset = make_know_pred(args, our_best_model, tokenizer, train_Dataset, knowledge_index_rerank, all_knowledgeDB)
        know_aug_test_dataset = make_know_pred(args, our_best_model, tokenizer, test_Dataset, knowledge_index_rerank, all_knowledgeDB)
        train_Dataset = Rag_context_Dataset(args, know_aug_train_dataset, rag_tokenizer, all_knowledgeDB, mode='train')
        test_Dataset = Rag_context_Dataset(args, know_aug_test_dataset, rag_tokenizer, all_knowledgeDB, mode='train')
        logger.info(f"Dataset Knowledge Augmented Finish")
    train_dataloader = DataLoader(train_Dataset, batch_size=args.rag_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_Dataset, batch_size=args.rag_batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(rag_model.parameters(), lr=args.rag_lr, weight_decay=0.1, eps=5e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
    best_hitdic_ratio = {'total': {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'total': 0}}
    best_hitdic_str = None
    logger.info(f"Logging Epoch results:                      hit@1, hit@3, hit@5, hit_new@1, hit_new@3, hit_new@5")
    
    train_dataloader = DataLoader(train_Dataset, batch_size=args.rag_batch_size, shuffle=True)
    if args.rag_our_model: play = epoch_play_by_context_input_ids
    else: play = epoch_play

    with torch.no_grad(): 
        play(args, rag_tokenizer, rag_model, test_dataloader, optimizer, scheduler, 0, faiss_dataset, mode='knowledgeCheckInTestDataset')
        # epoch_play_by_context_input_ids(args, rag_tokenizer, rag_model, test_dataloader, optimizer, scheduler, 0, faiss_dataset, mode='knowledgeCheck')
        # epoch_play(args, rag_tokenizer, rag_model, test_dataloader, optimizer, scheduler, 0, faiss_dataset, mode='knowledgeCheck')

    for epoch in range(args.rag_epochs):
        logger.info(f"RAG_LR: {args.rag_lr}")
        rag_model.train()
        if args.rag_onlyDecoderTune:
            logger.info(f"\n\n*****RAG_Only_Decoder Tune!***** rag_lr: {args.rag_lr}");
            logger.info(f"*****RAG_Only_Decoder Tune!***** rag_lr: {args.rag_lr}\n\n")
            rag_model.eval()
            if rag_model.rag.ctx_encoder: rag_model.rag.ctx_encoder.eval()
            rag_model.rag.question_encoder.eval()
            rag_model.generator.train()
            if rag_model.rag.ctx_encoder: 
                for param in rag_model.rag.ctx_encoder.parameters():
                    param.requires_grad = False
            for param in rag_model.rag.question_encoder.parameters():
                param.requires_grad = False
        if epoch == 0: rag_model_weight_logging(args, rag_model, epoch, 'before_train', faiss_dataset)
        hitDic, hitdic_ratio, output_str = play(args, rag_tokenizer, rag_model, train_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode = 'train')
        # hitDic, hitdic_ratio, output_str = epoch_play_by_context_input_ids(args, rag_tokenizer, rag_model, train_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode = 'train')
        # hitDic, hitdic_ratio, output_str = epoch_play(args, rag_tokenizer, rag_model, train_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode = 'train')

        rag_model.eval()
        with torch.no_grad():
            hitDic, hitdic_ratio, output_str = play(args, rag_tokenizer, rag_model, test_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode='test')
            # hitDic, hitdic_ratio, output_str = epoch_play_by_context_input_ids(args, rag_tokenizer, rag_model, test_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode='test')
            # hitDic, hitdic_ratio, output_str = epoch_play(args, rag_tokenizer, rag_model, test_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode='test')
            if best_hitdic_ratio['total']['hit1'] <= hitdic_ratio['total']['hit1']:
                best_hitdic_ratio = hitdic_ratio
                best_hitdic_str = output_str
        if epoch == 0: rag_model_weight_logging(args, rag_model, epoch, 'after_test', faiss_dataset)

    for i in best_hitdic_str:
        logger.info(f"Test_best {i}")

def epoch_play(args, tokenizer, model, data_loader, optimizer, scheduler, epoch, faiss_dataset, mode='train'):
    logger.info("RAG 가 직접 retrieve함")
    from tqdm import tqdm
    # data_loader
    epoch_loss, steps, gradient_accumulation_steps = 0, 0, 500
    torch.cuda.empty_cache()
    contexts, label_gold_knowledges, label_pseudo_knowledges, top5_docs, real_resps, gen_resp, new_knows = [], [], [], [], [], [], []
    types = []
    evaluatortype = ConvEvaluator_ByType(tokenizer=tokenizer, log_file_path=os.path.join(args.output_dir, f"{epoch}_{mode}_GEN_REPORT_TYPE.txt") if mode=='test' else None)

    for batch in tqdm(data_loader, desc=f"Epoch {epoch}__{mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["response"].to(args.device)
        ### lm_labels = target_ids # response == target_ids ### decoder_input_ids = target_ids[:, :-1].contiguous() ### lm_labels = target_ids[:, 1:].clone()  # decoder_input_ids = decoder_input_ids,

        #### Whole Model 사용시
        outputs = model(input_ids=source_ids,
                        attention_mask=source_mask,
                        labels=target_ids,  # target_ids = response
                        output_retrieved=True,
                        n_docs=5,
                        #### reduce_loss=True,       # HJ추가
                        #### exclude_bos_score=True, # HJ추가
                        )
        retrieved_docs_pt = outputs.retrieved_doc_ids.data

        loss = outputs['loss'].mean()
        epoch_loss += loss.item()
        # perplexity(outputs['logits'][::5].size(), target_ids) ## Perplexity 관련 코드
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            # if (steps+1) % gradient_accumulation_steps==0: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss.detach()
        steps += 1
        knowledge_gold_label = batch['target_knowledge_label']
        # knowledge_pseudo_label = batch['knowledge_task_pseudo_label']
        batch_types = [args.goalDic['int'][int(idx)] for idx in batch['goal_idx']]

        batch_top5_docs = [faiss_dataset[i]['text'] for i in retrieved_docs_pt]
        top5_docs.extend(batch_top5_docs)
        # new_knows.extend([int(i) for i in batch['is_new_knowledge']])
        contexts.extend(tokenizer.question_encoder.batch_decode(source_ids))
        real_resps.extend(tokenizer.generator.batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        label_gold_knowledges.extend(knowledge_gold_label)
        # label_pseudo_knowledges.extend(knowledge_pseudo_label)
        types.extend(batch_types)

        if mode == 'test':
            gen_ids = model.generate(source_ids, min_length=0, max_length=args.rag_max_target_length, early_stopping=True,
                                     num_beams=1, num_return_sequences=1, n_docs=5)
            resp_batch = tokenizer.generator.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_resp.extend(resp_batch)
            # evaluator.log_file.write(f'\n*** Generator Fine-tuning test-{epoch}{mode} ***\n\n')
            # evaluator.evaluate(gen_ids, target_ids, source_ids, log=True)
            # evaluator.evaluate(gen_ids, target_ids, log=True)
            evaluatortype.evaluate(gen_ids, target_ids, batch_types, log=True)

    if mode == 'train': scheduler.step()
    perplexity = torch.exp(torch.tensor(epoch_loss / steps))  # perplexity(outputs['logits'][::5], target_ids)
    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    if (epoch==0 and mode=='train') or 'knowledge' in mode:
        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
    if mode == 'test':
        # report = evaluator.report()
        # report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
        #                f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        # output_str.extend(report_text)
        report = evaluatortype.report()
        report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                       f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        output_str.extend(report_text)
        report_type = evaluatortype.report_ByType()
        for each_type, report in report_type.items():
            reports_text = [f"NEW_{epoch}_{mode:^5}_{each_type:^21}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4, count",
                            f"NEW_{epoch}_{mode:^5}_{each_type:^21}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}, Count: {report['sent_cnt']}",]
            output_str.extend(reports_text)
        
        # evaluator.reset_metric()
        evaluatortype.reset_metric()

        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
        logger.info(report_text[0])
        logger.info(report_text[1])
        logger.info("======------------============------------============------------============------------============------------======")
        bleu, bleu1, bleu2 = get_bleu(real_resps, gen_resp)
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(gen_resp)
        logger.info(f"                    PPL, Bleu_score, Bleu_1, Bleu_2: {perplexity:.3f}, {bleu:.3f}, {bleu1:.3f}, {bleu2:.3f}")
        logger.info(f"intra_dist1, intra_dist2, inter_dist1, inter_dist2 : {intra_dist1:.3f}, {intra_dist2:.3f}, {inter_dist1:.3f}, {inter_dist2:.3f}")
        output_str.append(f"PPL, Bleu_score, Bleu_1, Bleu_2                    : {perplexity:.3f}, {bleu:.3f}, {bleu1:.3f}, {bleu2:.3f}")
        output_str.append(f"intra_dist1, intra_dist2, inter_dist1, inter_dist2 : {intra_dist1:.3f}, {intra_dist2:.3f}, {inter_dist1:.3f}, {inter_dist2:.3f}")
        utils.write_pkl({'contexts':contexts, 'real_resp': real_resps, 'gen_resp': gen_resp, 'top5_docs':top5_docs, 'label_gold_knowledges':label_gold_knowledges, 'types': types}, os.path.join(args.output_dir,f"{epoch}_{mode}_inout.pkl"))
    logger.info(f"{mode} Loss: {epoch_loss:.3f}, PPL: {perplexity:.3f}")
    save_preds(args, contexts, top5_docs, label_gold_knowledges, epoch=epoch, new_knows=new_knows, real_resp=real_resps, gen_resps=gen_resp, mode=mode)
    return hitdic, hitdic_ratio, output_str  # output_strings, hit1_ratio, total_hit1, total_hit3, total_hit5, total_hit1_new, total_hit3_new, total_hit5_new


def epoch_play_by_context_input_ids(args, tokenizer, model, data_loader, optimizer, scheduler, epoch, faiss_dataset, mode='train'):
    logger.info("Retrieve된 input으로 받아서 생성 (context_input_ids)")
    from tqdm import tqdm
    epoch_loss, steps, gradient_accumulation_steps , cleanup = 0, 0, 500, False if epoch==0 else False
    torch.cuda.empty_cache()
    rag_contexts, contexts, label_gold_knowledges, label_pseudo_knowledges, top5_docs, real_resps, gen_resp, new_knows = [], [], [], [], [], [], [],[]
    types = []
    evaluatortype = ConvEvaluator_ByType(tokenizer=tokenizer, log_file_path=os.path.join(args.output_dir, f"{epoch}_{mode}_GEN_REPORT_TYPE.txt") if mode=='test' else None)
    for batch in tqdm(data_loader, desc=f"Epoch {epoch}__{mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["response"].to(args.device)
        ### lm_labels = target_ids # response == target_ids ### decoder_input_ids = target_ids[:, :-1].contiguous() ### lm_labels = target_ids[:, 1:].clone()  # decoder_input_ids = decoder_input_ids,
        # batch["context_input_ids"].reshape(-1, args.rag_max_input_length), batch["context_doc_scores"]
        #### Whole Model 사용시
        outputs = model(
                    context_input_ids = batch["context_input_ids"].reshape(-1, args.rag_context_input_length).to(args.device)
                    ,context_attention_mask = batch["context_input_attention_mask"].reshape(-1,args.rag_context_input_length).to(args.device)
                    ,decoder_input_ids = batch["response"].to(args.device)
                    ,doc_scores = batch['context_doc_scores'].to(args.device)
                    , labels = target_ids
                )

        loss = outputs['loss'].mean()
        epoch_loss += loss.item()
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            # if (steps+1) % gradient_accumulation_steps==0: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss.detach()
        steps += 1
        knowledge_gold_label = batch['target_knowledge_label']
        # knowledge_pseudo_label = batch['knowledge_task_pseudo_label']
        batch_types = [args.goalDic['int'][int(idx)] for idx in batch['goal_idx']]

        # batch_top5_docs = [faiss_dataset[i]['text'] for i in retrieved_docs_pt]
        # top5_docs.extend(batch_top5_docs)
        top5_docs.extend([[args.all_knowledgeDB[int(j)] for j in i] for i in batch['context_knowledges']]) # [[int(j) for j in i] for i in batch['context_knowledges'].detach()]
        # new_knows.extend([int(i) for i in batch['is_new_knowledge']])
        contexts.extend(tokenizer.question_encoder.batch_decode(source_ids))
        real_resps.extend(tokenizer.generator.batch_decode(target_ids, skip_special_tokens=cleanup, clean_up_tokenization_spaces=cleanup))
        label_gold_knowledges.extend(knowledge_gold_label)
        # label_pseudo_knowledges.extend(knowledge_pseudo_label)
        types.extend(batch_types)
        rag_contexts.extend(tokenizer.generator.batch_decode([i[0] for i in batch["context_input_ids"]]))
        if mode == 'test':
            gen_ids = model.generate(
                        context_input_ids = batch["context_input_ids"].reshape(-1, args.rag_context_input_length).to(args.device)
                        ,context_attention_mask = batch["context_input_attention_mask"].reshape(-1,args.rag_context_input_length).to(args.device)
                        ,doc_scores = batch['context_doc_scores'].to(args.device)
                        ,num_beams=1, early_stopping=True
                    )
            
            # model.generate(source_ids, min_length=0, max_length=args.rag_max_target_length, early_stopping=True,
            #                          num_beams=1, num_return_sequences=1, n_docs=5)
            resp_batch = tokenizer.generator.batch_decode(gen_ids, skip_special_tokens=cleanup, clean_up_tokenization_spaces=cleanup)
            gen_resp.extend(resp_batch)
            # evaluator.log_file.write(f'\n*** Generator Fine-tuning test-{epoch}{mode} ***\n\n')
            # evaluator.evaluate(gen_ids, target_ids, source_ids, log=True)
            # evaluator.evaluate(gen_ids, target_ids, log=True)
            evaluatortype.evaluate(gen_ids, target_ids, batch_types, log=True)

    if mode == 'train': scheduler.step()
    perplexity = torch.exp(torch.tensor(epoch_loss / steps))  # perplexity(outputs['logits'][::5], target_ids)
    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    if (epoch==0 and mode=='train') or 'knowledge' in mode:
        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
    if mode == 'test':
        # report = evaluator.report()
        # report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
        #                f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        # output_str.extend(report_text)
        report = evaluatortype.report()
        report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                       f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        output_str.extend(report_text)
        report_type = evaluatortype.report_ByType()
        for each_type, report in report_type.items():
            reports_text = [f"NEW_{epoch}_{mode:^5}_{each_type:^21}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4, count",
                            f"NEW_{epoch}_{mode:^5}_{each_type:^21}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}, Count: {report['sent_cnt']}",]
            output_str.extend(reports_text)
        
        # evaluator.reset_metric()
        evaluatortype.reset_metric()

        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
        logger.info(report_text[0])
        logger.info(report_text[1])
        logger.info("======------------============------------============------------============------------============------------======")
        # bleu, bleu1, bleu2 = get_bleu(real_resps, gen_resp)
        # intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(gen_resp)
        # logger.info(f"                    PPL, Bleu_score, Bleu_1, Bleu_2: {perplexity:.3f}, {bleu:.3f}, {bleu1:.3f}, {bleu2:.3f}")
        # logger.info(f"intra_dist1, intra_dist2, inter_dist1, inter_dist2 : {intra_dist1:.3f}, {intra_dist2:.3f}, {inter_dist1:.3f}, {inter_dist2:.3f}")
        # output_str.append(f"PPL, Bleu_score, Bleu_1, Bleu_2                    : {perplexity:.3f}, {bleu:.3f}, {bleu1:.3f}, {bleu2:.3f}")
        # output_str.append(f"intra_dist1, intra_dist2, inter_dist1, inter_dist2 : {intra_dist1:.3f}, {intra_dist2:.3f}, {inter_dist1:.3f}, {inter_dist2:.3f}")
        utils.write_pkl({'contexts':contexts, 'real_resp': real_resps, 'gen_resp': gen_resp, 'top5_docs':top5_docs, 'label_gold_knowledges':label_gold_knowledges, 'types': types}, os.path.join(args.output_dir,f"{epoch}_{mode}_inout.pkl"))
    logger.info(f"{mode} Loss: {epoch_loss:.3f}, PPL: {perplexity:.3f}")
    save_preds(args, contexts, top5_docs, label_gold_knowledges, epoch=epoch, new_knows=new_knows, real_resp=real_resps, gen_resps=gen_resp, mode=mode, rag_context=rag_contexts)
    return hitdic, hitdic_ratio, output_str  # output_strings, hit1_ratio, total_hit1, total_hit3, total_hit5, total_hit1_new, total_hit3_new, total_hit5_new





def know_hit_ratio(args, pred_pt, gold_pt, new_knows=None, types=None, typelist=['Q&A', 'Movie recommendation', 'Music recommendation', 'POI recommendation', 'Food recommendation']):
    if args.version=='ko': typelist = ['QA','Movie Recommendation']
    hitdic = {type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'total': 0} for type in typelist + ['Others', 'total']}
    for idx in range(len(gold_pt)):
        goal_type = types[idx]
        if goal_type in typelist:
            tmp_goal = goal_type
        else:
            tmp_goal = 'Others'

        pred, gold = pred_pt[idx], gold_pt[idx]

        hitdic[tmp_goal]['total'] += 1
        hitdic['total']['total'] += 1

        if args.rag_num_beams > 1:  
            if gold in pred:
                hitdic[tmp_goal]['hit5'] += 1
                hitdic['total']['hit5'] += 1
                if gold in pred[:3]:
                    hitdic[tmp_goal]['hit3'] += 1
                    hitdic['total']['hit3'] += 1
                    if gold == pred[0]:
                        hitdic[tmp_goal]['hit1'] += 1
                        hitdic['total']['hit1'] += 1
        else:
            if gold == pred: hitdic[tmp_goal]['hit1'] += 1
        if new_knows:
            new = new_knows[idx]
            if args.rag_num_beams > 1:
                if new and gold == pred[0]: hitdic[tmp_goal]['hit1_new'] += 1
                if new and gold in pred[:3]: hitdic[tmp_goal]['hit3_new'] += 1
                if new and gold in pred: hitdic[tmp_goal]['hit5_new'] += 1
            else:
                if new and gold == pred: hitdic[tmp_goal]['hit1_new'] += 1

    hitdic_ratio = {goal_type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'total': 0} for goal_type in typelist + ["Others", 'total']}
    output_str = [f"                         hit1,  hit3,  hit5, hit1_new, hit3_new, hit5_new, total_cnt"]
    for key in hitdic.keys():
        for hit in ['hit1', 'hit3', 'hit5']:
            if hitdic[key]['total']:
                hitdic_ratio[key][hit] = hitdic[key][hit] / hitdic[key]['total']
        hitdic_ratio[key]['total'] = hitdic[key]['total']
        output_str.append(f"{key:^22}: {hitdic_ratio[key]['hit1']:.3f}, {hitdic_ratio[key]['hit3']:.3f}, {hitdic_ratio[key]['hit5']:.3f}, {hitdic_ratio[key]['total']}")
    return hitdic, hitdic_ratio, output_str


def rag_model_weight_logging(args, model, epoch, mode, faiss_dataset):
    # weight_log_file = os.path.join(args.output_dir,f'{epoch}_{mode}_weights.txt')
    weight_log_file = os.path.join(args.output_dir, f'{epoch}_weights.txt')
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    with open(weight_log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{args.log_name}\n")
        f.write(f"\n only decoder tune: {args.rag_onlyDecoderTune} // rag_our_model: {args.rag_our_model}\n")
        f.write(f"{epoch}_{mode}\n")
        f.write(f"model.question_encoder.training: {model.question_encoder.training}\n")
        f.write(f"model.generator.training: {model.generator.training}\n")
        f.write(f"model.rag.training: {model.rag.training}\n")
        f.write(f"model.rag.generator.training: {model.rag.generator.training}\n")
        if model.rag.ctx_encoder:
            f.write(f"model.rag.ctx_encoder.training: {model.rag.ctx_encoder.training}\n")
            f.write(f"\nmodel.rag.ctx_encoder.ctx_encoder.bert_model.encoder.layer[0].attention.self.key.weight[0][:50][0]\n")
            f.write(f'{model.rag.ctx_encoder.ctx_encoder.bert_model.encoder.layer[0].attention.self.key.weight[0][:50][0]}\n')
        f.write(f"\nmodel.rag.question_encoder.question_encoder.bert_model.base_model.encoder.layer[0].attention.self.key.weight[0][:50][0]\n")
        f.write(f"{model.rag.question_encoder.question_encoder.bert_model.base_model.encoder.layer[0].attention.self.key.weight[0][:50][0]}")
        f.write(f"\nmodel.rag.generator.model.encoder.base_model.layers[0].self_attn.k_proj.weight[0][:50][0]\n")
        f.write(f"{model.rag.generator.model.encoder.base_model.layers[0].self_attn.k_proj.weight[0][:50][0]}")
        f.write(f"\nmodel.rag.generator.model.decoder.base_model.layers[0].self_attn.k_proj.weight[0][:50][0]\n")
        f.write(f'{model.rag.generator.model.decoder.base_model.layers[0].self_attn.k_proj.weight[0][:50][0]}\n')
        f.write(f"\nfaiss dataset [0,5,10,15,20][:50][0]\n")
        f.write(f'{faiss_dataset[0]["embeddings"][:50][0]}\n')
        f.write(f'{faiss_dataset[5]["embeddings"][:50][0]}\n')
        f.write(f'{faiss_dataset[10]["embeddings"][:50][0]}\n')
        f.write(f'{faiss_dataset[15]["embeddings"][:50][0]}\n')
        f.write(f'{faiss_dataset[20]["embeddings"][:50][0]}\n')
        f.write(f'{mode}-----------------End----------------\n\n')


def save_preds(args, context, pred_words, label_words, epoch=None, new_knows=None, real_resp=None, gen_resps=None, mode='train', rag_context=None):
    # HJ: 동일 파일 덮어쓰면서 맨 윗줄에 몇번째 에폭인지만 쓰도록 수정
    log_file_name = mode + f'{str(epoch)}_' + args.log_name
    path = os.path.join(args.output_dir, log_file_name)
    # if not os.path.exists(path): os.makedirs(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{mode}, Epoch: {str(epoch)} Input and Output results {args.time}\n")
        f.write(f"Log File Name: {args.log_name} \n")
        for i, (ctx, pred, label) in enumerate(zip(context, pred_words, label_words)):
            if i == 500: break
            f.write(f"Source: {ctx}\n")
            if rag_context: f.write(f"Context_Input Top1: {rag_context[i]}\n")
            if new_knows: f.write(f"Is_New_Knows: {new_knows[i]}\n")
            f.write(f"Pred  : {pred}\n")
            f.write(f"Label : {label}\n")
            f.write(f"Real Response: {real_resp[i]}\n")
            if gen_resps: f.write(f"Gen Response : {gen_resps[i]}\n")
            f.write(f"\n")
    logger.info(f"Save {mode}, Epoch: {str(epoch)}, generated results in {path}")
    return


def get_bleu(real_resps, predicts):  # From UNIMIND
    ref = [[gold.lower().split(' ')] for gold in real_resps]  # len of Golden samples of [ [['System:', "It's", 'Libra.[SEP]']], ... ,[['System:', "Okay", 'Goodbye[SEP]']] ]
    preds = [pred.lower().split(' ') for pred in predicts]  # len of Predicted samples of [ ['yeah','i','think','so',], ... , ['see','you'] ]
    # assert isinstance(ref[0][0], list) and isinstance(preds[0], list)
    bleu_score = corpus_bleu(ref, preds)
    bleu1 = corpus_bleu(ref, preds, weights=(1, 0, 0, 0))  # bleu 1-gram
    bleu2 = corpus_bleu(ref, preds, weights=(0.5, 0.5, 0, 0))  # bleu 2-gram
    return bleu_score, bleu1, bleu2


def distinct(candidates):  # From UniMIND
    seqs = [pred.lower().split(' ') for pred in candidates]
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams) + 1e-12) / (len(seq) + 1e-5))
        intra_dist2.append((len(bigrams) + 1e-12) / (max(0, len(seq) - 1) + 1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
    inter_dist2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)
    intra_dist1 = np.average(intra_dist1)  # Dist
    intra_dist2 = np.average(intra_dist2)  # Dist
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

from collections import defaultdict
import random

class Rag_context_Dataset(Dataset):
    def __init__(self, args, augmented_raw_sample, tokenizer=None, knowledgeDB=None, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.mode = mode
        self.tokenizer = tokenizer
        self.augmented_raw_sample = augmented_raw_sample
        self.input_max_length = args.rag_context_input_length     # TODO: TEMP args.rag_max_input_length=args.rag_context_input_length
        self.target_max_length = args.rag_max_target_length   # TODO: TEMP
        self.knowledgeDB = knowledgeDB

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        data = self.augmented_raw_sample[item]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        dialog, response = dialog.replace('[SEP]','</s>') , response.replace('[SEP]','</s>')
        pad_token_id = self.tokenizer.question_encoder.pad_token_id

        context_batch = defaultdict()
        predicted_topic_list = deepcopy(data['predicted_topic'][:self.args.topk_topic])
        predicted_topic_confidence_list = deepcopy(data['predicted_topic_confidence'][:self.args.topk_topic])

        if self.mode == 'train':
            random.shuffle(predicted_topic_list)
            predicted_goal, predicted_topic = data['predicted_goal'][0], '|'.join(predicted_topic_list)
        else:  # test
            cum_prob = 0
            candidate_topic_entities = []
            for topic, conf in zip(predicted_topic_list, predicted_topic_confidence_list):
                candidate_topic_entities.append(topic)
                cum_prob += conf
                if cum_prob > self.args.topic_conf:
                    break
            predicted_goal, predicted_topic = data['predicted_goal'][0],  '|'.join(candidate_topic_entities)

        if self.args.rag_our_model == 'DPR' or self.args.rag_our_model == 'dpr': prefix = ''
        elif self.args.rag_our_model == 'C2DPR' or self.args.rag_our_model == 'c2dpr': prefix = '<topic>' + predicted_topic + self.tokenizer.question_encoder.sep_token
        else: prefix = '' # Scratch DPR
            


        prefix_encoding = self.tokenizer.question_encoder.encode(prefix)[1:-1][:self.input_max_length//4]  # --> 64까지 늘어나야함
        input_sentence = self.tokenizer.question_encoder('<dialog>' + dialog, add_special_tokens=False).input_ids
        input_sentence = [self.tokenizer.question_encoder.cls_token_id] + prefix_encoding + input_sentence[-(self.input_max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.input_max_length - len(input_sentence))

        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        # response에서 [SEP] token 제거

        if '[SEP]' in response: response = response[: response.index("[SEP]")]

        labels = self.tokenizer.generator(response, max_length=self.target_max_length, padding='max_length', truncation=True)['input_ids']

        ## Context_input_ids 사용하기 Start ##
        if self.args.rag_our_model:
            top5_knows, top5_confs = data['predicted_know'], data['predicted_know_confidence'] # candidate_knowledges[:5], candidate_knowledges[:5]]
            context_batch['context_input_ids']=[]
            context_batch['context_input_attention_mask']=[]
            context_batch['context_doc_scores'] =[]
            context_batch['context_knowledges'] =[]
            # if self.args.rag_our_model
            #  f"{dialog} goal: {predicted_goal} topic: {predicted_topic} Generate the response: "
            if self.args.rag_context_input_only_dialog_doc: context_input5 = [f"<s>{pred_know}</s>{dialog}</s>"  for pred_know in top5_knows]
            else:  context_input5 = [f"<s>{pred_know}</s>{dialog} goal: {predicted_goal} </s> topic: {predicted_topic} Generate the response: </s>"  for pred_know in top5_knows]
            context_input5_tokenizes = self.tokenizer.generator(context_input5, max_length=self.input_max_length, padding='max_length', truncation=True)
            doc_scores = candidate_confidences[:5]
            for context_input5_input_ids,context_input5_attention_masks, doc_score in zip(context_input5_tokenizes.input_ids, context_input5_tokenizes.attention_mask, doc_scores):
                context_batch['context_input_ids'].append(context_input5_input_ids)
                context_batch['context_input_attention_mask'].append(context_input5_attention_masks)
                context_batch['context_doc_scores'].append(doc_score)
                context_batch['context_knowledges'] = [self.args.all_knowledgeDB.index(i) for i in top5_knows]
        ## END ##



        context_batch['response'] = [self.tokenizer.generator.bos_token_id] + labels # kobart <s> issue
        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        # context_batch['topic'] = self.tokenizer(topic, truncation=True, padding='max_length', max_length=32).input_ids
        # context_batch['target_knowledge_label'] = self.knowledgeDB.index(target_knowledge)
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        context_batch['target_knowledge_label'] = target_knowledge.replace('\t', ' ')
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)

        
class RagDataset(Dataset):
    def __init__(self, args, augmented_raw_sample, tokenizer=None, knowledgeDB=None, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.mode = mode
        self.tokenizer = tokenizer
        self.augmented_raw_sample = augmented_raw_sample
        self.input_max_length = 128# args.rag_max_input_length
        self.target_max_length = 128# args.rag_max_target_length
        self.knowledgeDB = knowledgeDB

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        data = self.augmented_raw_sample[item]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]

        pad_token_id = self.tokenizer.question_encoder.pad_token_id

        context_batch = defaultdict()
        predicted_topic_list = deepcopy(data['predicted_topic'][:self.args.topk_topic])
        predicted_topic_confidence_list = deepcopy(data['predicted_topic_confidence'][:self.args.topk_topic])

        if self.mode == 'train':
            random.shuffle(predicted_topic_list)
            predicted_goal, predicted_topic = data['predicted_goal'][0], '|'.join(predicted_topic_list)
        else:  # test
            cum_prob = 0
            candidate_topic_entities = []
            for topic, conf in zip(predicted_topic_list, predicted_topic_confidence_list):
                candidate_topic_entities.append(topic)
                cum_prob += conf
                if cum_prob > self.args.topic_conf: break
            predicted_topic = '|'.join(candidate_topic_entities)

        if self.args.rag_our_model == 'DPR' or self.args.rag_our_model == 'dpr':
            prefix = ''
        elif self.args.rag_our_model == 'C2DPR' or self.args.rag_our_model == 'c2dpr':
            prefix = '<topic>' + predicted_topic + self.tokenizer.question_encoder.sep_token
        else:  # Scratch DPR
            prefix = ''

        prefix_encoding = self.tokenizer.question_encoder.encode(prefix)[1:-1][:self.input_max_length//4]  # --> 64까지 늘어나야함

        input_sentence = self.tokenizer.question_encoder('<dialog>' + dialog, add_special_tokens=False).input_ids
        input_sentence = [self.tokenizer.question_encoder.cls_token_id] + prefix_encoding + input_sentence[-(self.input_max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.input_max_length - len(input_sentence))

        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        # response에서 [SEP] token 제거

        if '[SEP]' in response: response = response[: response.index("[SEP]")]

        labels = self.tokenizer.generator(response, max_length=self.target_max_length, padding='max_length', truncation=True)['input_ids']

        context_batch['response'] = [self.tokenizer.generator.bos_token_id] + labels
        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        # context_batch['topic'] = self.tokenizer(topic, truncation=True, padding='max_length', max_length=32).input_ids
        # context_batch['target_knowledge_label'] = self.knowledgeDB.index(target_knowledge)
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        context_batch['target_knowledge_label'] = target_knowledge.replace('\t', ' ')
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)

# def split_documents(documents: dict) -> dict:
#     """Split documents into passages"""
#     titles, texts = [], []
#     for title, text in zip(documents["title"], documents["text"]):
#         if text is not None:
#             for passage in split_text(text):
#                 titles.append(title if title is not None else "")
#                 texts.append(passage)
#     return {"title": titles, "text": texts}


# def split_text(text: str, n=100, character=" ") -> List[str]:
#     """Split the text every ``n``-th occurrence of ``character``"""
#     text = text.split(character)
#     return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]
