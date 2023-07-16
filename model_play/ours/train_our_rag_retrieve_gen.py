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
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagSequenceForGeneration, RagTokenizer
from typing import List
from datasets import Features, Sequence, Value, load_dataset
from functools import partial
import faiss
from models.ours.retriever import Retriever
import utils
import data_model


def train_our_rag_generation(args, bert_model, tokenizer, all_knowledgeDB):
    logger.info("OUR Retriever model For resp")
    from model_play.rag import rag_retrieve
    # if args.rag_onlyDecoderTune: args.rag_batch_size = args.rag_batch_size*2

    our_best_model = Retriever(args, bert_model)
    our_best_model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"ours_retriever.pt"), map_location=args.device))
    our_best_model.to(args.device)
    our_question_encoder = our_best_model.query_bert
    our_ctx_encoder = our_best_model.rerank_bert

    knowledgeDB_list = list(all_knowledgeDB)
    knowledgeDB_csv_path = os.path.join(args.data_dir, 'rag')  # HOME/data/2/rag/"train_knowledge.csv")
    utils.checkPath(knowledgeDB_csv_path)
    knowledgeDB_csv_path = os.path.join(knowledgeDB_csv_path, f'my_knowledge_dataset_{args.gpu}' + ('_debug.csv' if args.debug else '.csv'))
    args.knowledgeDB_csv_path = knowledgeDB_csv_path
    with open(knowledgeDB_csv_path, 'w', encoding='utf-8') as f:
        for know in knowledgeDB_list:
            f.write(f" \t{know}\n")
    faiss_dataset = load_dataset("csv", data_files=[knowledgeDB_csv_path], split="train", delimiter="\t", column_names=["title", "text"])
    faiss_dataset = faiss_dataset.map(rag_retrieve.split_documents, batched=True, num_proc=4)

    MODEL_CACHE_DIR = os.path.join(args.home, 'model_cache', 'facebook/dpr-ctx_encoder-multiset-base')

    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR).to(device=args.device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=MODEL_CACHE_DIR)

    if args.rag_scratch:
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Use Our Trained Bert For ctx_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Use Our Trained Bert For ctx_encoder @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        ctx_encoder.ctx_encoder.bert_model = our_ctx_encoder
        ctx_tokenizer = tokenizer

    logger.info("Create Knowledge Dataset")
    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})  # optional, save as float32 instead of float64 to save space
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

    ### MODEL CALL
    retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='custom', indexed_dataset=faiss_dataset, init_retrieval=True)
    retriever.set_ctx_encoder_tokenizer(ctx_tokenizer) # NO TOUCH
    rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever).to(args.device)
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    rag_model.set_context_encoder_for_training(ctx_encoder)
    if args.rag_scratch:
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@ Model question_encoder changed by ours @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@ Model question_encoder changed by ours @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        rag_model.rag.question_encoder.question_encoder.bert_model = our_question_encoder
        rag_tokenizer.question_encoder = tokenizer

    train_aug_pred_path = os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl')
    test_aug_pred_path = os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl')
    assert os.path.exists(train_aug_pred_path) and os.path.exists(test_aug_pred_path), f"Goal,Topic Predicted file doesn't exist! {train_aug_pred_path}"
    
    train_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_train_pred_aug_dataset.pkl'))
    test_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'gt_test_pred_aug_dataset.pkl'))
    if args.debug: train_dataset_aug_pred , test_dataset_aug_pred = train_dataset_aug_pred[:50], test_dataset_aug_pred[:50]

    train_Dataset = data_model.RagDataset(args, train_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='train')
    test_Dataset = data_model.RagDataset(args, test_dataset_aug_pred, rag_tokenizer, all_knowledgeDB, mode='test')

    # train_our_rag_retrieve_gen.train_our_rag(args, rag_model, rag_tokenizer, faiss_dataset, train_Dataset, test_Dataset)

    # from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_Dataset, batch_size=args.rag_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_Dataset, batch_size=args.rag_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(rag_model.parameters(), lr=args.rag_lr, weight_decay=0.1, eps=5e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.rag_epochs * len(train_dataloader), eta_min= args.rag_lr * 0.1)
    best_hitdic_ratio = {'total': {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'total': 0}}
    best_hitdic_str = None
    logger.info(f"Logging Epoch results:                      hit@1, hit@3, hit@5, hit_new@1, hit_new@3, hit_new@5")
    early_stop_counter, early_stop = 0 , False
    for epoch in range(args.rag_epochs):
        # if not args.debug:
        rag_model.train()
        if args.rag_onlyDecoderTune:
            rag_model.eval()
            rag_model.generator.train()
            hitDic, hitdic_ratio, output_str = epoch_play(args, rag_tokenizer, rag_model, train_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode = 'train')

        rag_model.eval()
        with torch.no_grad():
            hitDic, hitdic_ratio, output_str = epoch_play(args, rag_tokenizer, rag_model, test_dataloader, optimizer, scheduler, epoch, faiss_dataset, mode='test')
            if best_hitdic_ratio['total']['hit1'] <= hitdic_ratio['total']['hit1']:
                best_hitdic_ratio = hitdic_ratio
                best_hitdic_str = output_str
            # else:
            #     early_stop+=1 # 3회 이상 hit1 ratio 떨어지면 그냥 꺼버리기

        # if epoch<3: 
        #     index_update(args, rag_model, rag_tokenizer, faiss_dataset)

        ## Early Stopping


    for i in best_hitdic_str:
        logger.info(f"Test_best {i}")



def epoch_play(args, tokenizer, model, data_loader, optimizer, scheduler, epoch, faiss_dataset, mode='train'):
    from tqdm import tqdm
    # data_loader
    epoch_loss, steps = 0, 0
    torch.cuda.empty_cache()
    contexts, label_gold_knowledges, label_pseudo_knowledges, top5_docs, real_resps, gen_resp, new_knows = [], [], [], [], [], [], []
    types = []
    for batch in tqdm(data_loader, desc=f"Epoch {epoch}__{mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        source_ids, source_mask, target_ids = batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device), batch["response"].to(args.device)
        ### lm_labels = target_ids # response == target_ids ### decoder_input_ids = target_ids[:, :-1].contiguous() ### lm_labels = target_ids[:, 1:].clone()
        outputs = model(input_ids=source_ids,
                        attention_mask=source_mask,
                        labels=target_ids,  # target_ids = response
                        output_retrieved=True)
        # decoder_input_ids = decoder_input_ids,
        retrieved_docs_pt = outputs.retrieved_doc_ids.data
        loss = outputs['loss'].mean()
        epoch_loss += loss.item()
        # question_encoder.
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.detach()
        steps+=1
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
        
        if mode == 'test' :
            resp_batch = tokenizer.generator.batch_decode(
                model.generate(source_ids, min_length=0, max_length=args.rag_max_target_length, early_stopping=True,
                               num_beams=1, num_return_sequences=1, n_docs=5
                               )
                , skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_resp.extend(resp_batch)
    if mode=='train': scheduler.step()
    perplexity = torch.exp(torch.tensor(epoch_loss/steps))
    hitdic, hitdic_ratio, output_str = know_hit_ratio(args, pred_pt=top5_docs, gold_pt=label_gold_knowledges, new_knows=new_knows, types=types)
    if mode == 'test':
        for i in output_str:
            logger.info(f"{mode}_{epoch} {i}")
        bleu, bleu1, bleu2 = get_bleu(real_resps, gen_resp)
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(gen_resp)
        logger.info(f"PPL, Bleu_score, Bleu_1, Bleu_2: {perplexity:.3f}, {bleu:.3f}, {bleu1:.3f}, {bleu2:.3f}")
        logger.info(f"intra_dist1, intra_dist2, inter_dist1, inter_dist2 : {intra_dist1:.3f}, {intra_dist2:.3f}, {inter_dist1:.3f}, {inter_dist2:.3f}")
        output_str.append(f"PPL, Bleu_score, Bleu_1, Bleu_2: {perplexity:.3f}, {bleu:.3f}, {bleu1:.3f}, {bleu2:.3f}")
        output_str.append(f"intra_dist1, intra_dist2, inter_dist1, inter_dist2 : {intra_dist1:.3f}, {intra_dist2:.3f}, {inter_dist1:.3f}, {inter_dist2:.3f}")
    logger.info(f"{mode} Loss: {epoch_loss}, PPL: {perplexity}")
    save_preds(args, contexts, top5_docs, label_gold_knowledges, epoch=epoch, new_knows=new_knows, real_resp=real_resps, gen_resps=gen_resp, mode=mode)
    return hitdic, hitdic_ratio, output_str  # output_strings, hit1_ratio, total_hit1, total_hit3, total_hit5, total_hit1_new, total_hit3_new, total_hit5_new


def know_hit_ratio(args, pred_pt, gold_pt, new_knows=None, types=None, typelist=['Q&A', 'Movie recommendation', 'Music recommendation', 'POI recommendation', 'Food recommendation']):
    # TODO: Beam처리
    hitdic={type:{'hit1':0, 'hit3':0, 'hit5':0, 'hit1_new':0, 'hit3_new':0, 'hit5_new':0,  'total':0} for type in typelist + ['Others', 'total']}
    for idx in range(len(gold_pt)):
        goal_type=types[idx]
        if goal_type in typelist: tmp_goal=goal_type
        else: tmp_goal='Others'

        pred, gold = pred_pt[idx], gold_pt[idx]

        hitdic[tmp_goal]['total']+=1
        hitdic['total']['total']+=1

        if args.rag_num_beams>1:
            if gold in pred:
                hitdic[tmp_goal]['hit5']+=1
                hitdic['total']['hit5']+=1
                if gold in pred[:3]:
                    hitdic[tmp_goal]['hit3']+=1
                    hitdic['total']['hit3']+=1
                    if gold == pred[0]:
                        hitdic[tmp_goal]['hit1']+=1
                        hitdic['total']['hit1']+=1
        else:
            if gold==pred : hitdic[tmp_goal]['hit1']+=1
        if new_knows:
            new=new_knows[idx]
            if args.rag_num_beams>1:
                if new and gold == pred[0]: hitdic[tmp_goal]['hit1_new']+=1
                if new and gold in pred[:3]: hitdic[tmp_goal]['hit3_new']+=1
                if new and gold in pred: hitdic[tmp_goal]['hit5_new']+=1
            else:
                if new and gold==pred : hitdic[tmp_goal]['hit1_new']+=1

    hitdic_ratio = {goal_type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit1_new':0, 'hit3_new':0, 'hit5_new':0, 'total': 0} for goal_type in typelist + ["Others", 'total']}
    output_str = [f"                         hit1,  hit3,  hit5, hit1_new, hit3_new, hit5_new, total_cnt"]
    for key in hitdic.keys():
        for hit in ['hit1', 'hit3', 'hit5']:
            if hitdic[key]['total']:
                hitdic_ratio[key][hit] = hitdic[key][hit] / hitdic[key]['total']
        hitdic_ratio[key]['total'] = hitdic[key]['total']
        output_str.append(f"{key:^22}: {hitdic_ratio[key]['hit1']:.3f}, {hitdic_ratio[key]['hit3']:.3f}, {hitdic_ratio[key]['hit5']:.3f}, {hitdic_ratio[key]['total']}")
    # for key in hitdic.keys():
    #     hitdic_ratio[key]['total'] = hitdic[key]['total']
    #     if key=='total': continue
    #     for hit in ['hit1', 'hit3', 'hit5']:
    #         if hitdic[key]['total'] > 0:
    #             hitdic_ratio[key][hit] = hitdic[key][hit] / hitdic[key]['total']
    #     output_str.append(f"{key:^22}: {hitdic_ratio[key]['hit1']:.3f}, {hitdic_ratio[key]['hit3']:.3f}, {hitdic_ratio[key]['hit5']:.3f}, {hitdic_ratio[key]['total']}")
    # hitdic_ratio['total']['hit1'],hitdic_ratio['total']['hit3'], hitdic_ratio['total']['hit5'],
    # output_str.append(f"{'total':^22}: {hitdic_ratio['total']['hit1']/hitdic_ratio['total']['total']:.3f}, {hitdic_ratio['total']['hit3']/hitdic_ratio['total']['total']:.3f}, {hitdic_ratio['total']['hit5']/hitdic_ratio['total']['total']:.3f}, {hitdic_ratio['total']['total']}")
    return hitdic, hitdic_ratio, output_str


def save_preds(args, context, pred_words, label_words, epoch=None, new_knows=None, real_resp=None, gen_resps=None, mode='train'):
    # HJ: 동일 파일 덮어쓰면서 맨 윗줄에 몇번째 에폭인지만 쓰도록 수정
    log_file_name = mode + f'{str(epoch)}_'+ args.log_name
    path = os.path.join(args.output_dir, log_file_name)
    # if not os.path.exists(path): os.makedirs(path)
    with open(path , 'w' ,encoding='utf-8') as f:
        f.write(f"{mode}, Epoch: {str(epoch)} Input and Output results {args.time}\n")
        f.write(f"Log File Name: {args.log_name} \n")
        for i,(ctx, pred, label) in enumerate(zip(context, pred_words, label_words)):
            if i==500: break
            f.write(f"Source: {ctx}\n")
            if new_knows: f.write(f"Is_New_Knows: {new_knows[i]}\n")
            f.write(f"Pred : {pred}\n")
            f.write(f"Label: {label}\n")
            f.write(f"Real Response: {real_resp[i]}\n")
            if gen_resps: f.write(f"Gen Response: {gen_resps[i]}\n")
            f.write(f"\n")
    logger.info(f"Save {mode}, Epoch: {str(epoch)}, generated results in {path}")
    return

def get_bleu(real_resps, predicts): # From UNIMIND
    ref = [[gold.lower().split(' ')] for gold in real_resps] # len of Golden samples of [ [['System:', "It's", 'Libra.[SEP]']], ... ,[['System:', "Okay", 'Goodbye[SEP]']] ]
    preds = [pred.lower().split(' ') for pred in predicts] # len of Predicted samples of [ ['yeah','i','think','so',], ... , ['see','you'] ]
    # assert isinstance(ref[0][0], list) and isinstance(preds[0], list)
    bleu_score = corpus_bleu(ref, preds)
    bleu1 = corpus_bleu(ref, preds, weights=(1, 0, 0, 0)) # bleu 1-gram
    bleu2 = corpus_bleu(ref, preds, weights=(0.5, 0.5, 0, 0)) # bleu 2-gram
    return bleu_score, bleu1, bleu2

def distinct(candidates): # From UniMIND
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
    intra_dist1 = np.average(intra_dist1) # Dist
    intra_dist2 = np.average(intra_dist2) # Dist
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

def index_update(args, model=None, tokenizer=None, dataset=None):
    if model: ctx_encoder = model.rag.ctx_encoder
    else: ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=os.path.join(args.home,'model_cache')).to(device=args.device)
    # ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", cache_dir=os.path.join(args.home,'model_cache'))
    ctx_tokenizer = tokenizer
    # knowledgeDB_csv_path=os.path.join(args.home, 'data', 'rag', 'my_knowledge_dataset.csv')
    dataset = load_dataset("csv", data_files=[args.knowledgeDB_csv_path], split="train", delimiter="\t", column_names=["title", "text"])
    dataset = dataset.map(split_documents, batched=True, num_proc = 4)
    
    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})  # optional, save as float32 instead of float64 to save space
    logger.info("Create Knowledge Dataset")
    new_dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer, args=args),
        batched=True, batch_size = args.batch_size, features=new_features,)

    new_dataset.save_to_disk(args.passages_path)

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    new_dataset.add_faiss_index("embeddings", custom_index=index)
    # model.rag.retriever.re_load() # Error
    model.rag.retriever.init_retrieval()
    

def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}

def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]

def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast, args) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt")["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=args.device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}







# if __name__ == "__main__":
#     # args = parseargs()
#     import main
#     main.main()