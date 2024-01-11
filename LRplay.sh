#!/bin/bash
# 아래에 실행시키려는 녀석들 다 입력해놓고, 마지막 echo "" 따옴표 안에 어떤걸 보기위한 실험이었는지 적어놓기

# --train_on_inputs=False --system_template --do_not_create user's response 

CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --learning_rate=1e-4 --log_name=Llama_default_1e-4 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_origin 
CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --learning_rate=2e-4 --log_name=Llama_default_2e-4 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_origin 
CUDA_VISIBLE_DEVICES=2 python llama_main_finetune.py --learning_rate=4e-4 --log_name=Llama_default_4e-4 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_origin 
CUDA_VISIBLE_DEVICES=3 python llama_main_finetune.py --learning_rate=5e-4 --log_name=Llama_default_5e-4 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_origin 



# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=llama_DPROUR --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=dpr --idea=1_2
# CUDA_VISIBLE_DEVICES=2 python llama_main_finetune.py --log_name=llama_DPROUR --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=dpr --idea=1_2
# CUDA_VISIBLE_DEVICES=3 python llama_main_finetune.py --log_name=llama_DPR_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=dpr --idea=0
# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=llama_DPR_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=dpr --idea=0
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=llama_DPR_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=dpr --idea=0
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=llama_DPR_No_restart --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=dpr --idea=0 --lora_weights=2024-01-09_131541_llama_DPR_No_llama_log.txt_Epoch

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=llama_ContrieverOUR --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=contriever --idea=1_2
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=llama_ContrieverOUR --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=contriever --idea=1_2
# CUDA_VISIBLE_DEVICES=2 python llama_main_finetune.py --log_name=llama_ContrieverOUR --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=contriever --idea=1_2
# CUDA_VISIBLE_DEVICES=3 python llama_main_finetune.py --log_name=llama_Contriever_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=contriever --idea=0
# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=llama_Contriever_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=contriever --idea=0
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=llama_Contriever_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=contriever --idea=0
CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=RE_llama_Contriever_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=contriever --idea=0 --lore_weights=2024-01-09_204945_llama_Contriever_No_llama_log.txt_Epoch

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=llama_CotmaeOUR --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=cotmae --idea=1_2
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=llama_CotmaeOUR --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=cotmae --idea=1_2
# CUDA_VISIBLE_DEVICES=2 python llama_main_finetune.py --log_name=llama_CotmaeOUR --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=cotmae --idea=1_2
# CUDA_VISIBLE_DEVICES=3 python llama_main_finetune.py --log_name=llama_Cotmae_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=cotmae --idea=0
# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=llama_Cotmae_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=cotmae --idea=0
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=llama_Cotmae_No --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=cotmae --idea=0

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=llama_CotmaeOUR --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True --knowledge_method=cotmae --idea=1_2



# python pseudo_labeler.py --mode=test --how=dialog --gpu=0 --save --score_method=bm25  --log_name=BM25
# python pseudo_labeler.py --mode=test --how=resp_uttr_item --gpu=0 --save --score_method=bm25  --log_name=BM25_uttr_resp_item

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --learning_rate=3e-4 --log_name=Lm7bhf_len512_3e4_prompt_WithKnow_Token_0 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --learning_rate=3e-4 --log_name=Lm7bhf_len512_3e4_prompt_WithKnow_Token_1 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True
# CUDA_VISIBLE_DEVICES=2 python llama_main_finetune.py --learning_rate=3e-4 --log_name=Lm7bhf_len512_3e4_prompt_WithKnow_Token_2 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage_token --prompt_w_knowledge=True


# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --learning_rate=3e-4 --log_name=JW_Lm7bhf_len512_3e4_prompt_WithKnow_0 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage --prompt_w_knowledge=True
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --learning_rate=3e-4 --log_name=JW_Lm7bhf_len512_3e4_prompt_WithKnow_1 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage --prompt_w_knowledge=True
# CUDA_VISIBLE_DEVICES=2 python llama_main_finetune.py --learning_rate=3e-4 --log_name=JW_Lm7bhf_len512_3e4_prompt_WithKnow_2 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_w_passage --prompt_w_knowledge=True


# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --learning_rate=3e-4 --log_name=JW_Lm7bhf_len512_3e4_prompt_With샵샵샵Resp_2 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_origin 

# CUDA_VISIBLE_DEVICES=2 python llama_main_finetune.py --learning_rate=3e-4 --log_name=JW_Lm7bhf_len512_3e4_prompt_No샵샵샵Resp --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_1 
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --learning_rate=3e-4 --log_name=JW_Lm7bhf_len512_3e4_prompt_With샵샵샵Resp --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_origin 


# CUDA_VISIBLE_DEVICES=2 python llama_main_finetune.py --log_name=Plz_L7b_chat_len512 --epoch=5 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --epoch=7
# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --learning_rate=3e-4 --log_name=Lm7bhf_len512_3e4_prompt_System_NoSEP --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_system 
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --learning_rate=3e-4 --log_name=Lm7bhf_len512_3e4_prompt1_NoSEP --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_1



# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --learning_rate=3e-4 --log_name=7bhf_len512_3e4_prompt3 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_3
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --learning_rate=3e-4 --log_name=Llama7b_chathf_len512_3e4_prompt0 --llama_input_maxlen=512 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_0

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --learning_rate=2e-4 --log_name=Llama7b_chathf_len256_2e4 --llama_input_maxlen=256 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=test --prompt=template_1 --lora_weights=2024-01-02_200254_Llama7b_chathf_len256_2e4_llama_log.txt_Epoch

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --learning_rate=2e-4 --log_name=Llama7b_chathf_len256_2e4 --llama_input_maxlen=256 --epoch=5 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_1
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --learning_rate=4e-4 --log_name=Llama7b_chathf_len512_4e4 --llama_input_maxlen=512 --epoch=5 --base_model=meta-llama/Llama-2-7b-chat-hf --mode=train_test --prompt=template_1 # 터짐


# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=Plz_Llama7b_chat_len512_2023-12-09_143515_Test --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=test --lora_weights=2023-12-09_143515_7b_len512_promptHJ_3e4_llama_log.txt_Epoch

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=Plz_Llama7b_chat_len512_2023-12-07_235521_len512_Epoch5_Test --epoch=5 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=test --lora_weights=2023-12-07_235521_Llama7b_chat_len512_llama_log.txt_Epoch5
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=Plz_Llama7b_chat_len512_2023-12-07_235521_len512_Epoch5_Test_len256으로 --epoch=5 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=256 --mode=test --lora_weights=2023-12-07_235521_Llama7b_chat_len512_llama_log.txt_Epoch5

# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=0 --kers_input_length=256 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=2 --lr=1e-5 --log_name="Resp_init_2layer_lr1e5_KnowFromKERS_cand20_Input256" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=0 --kers_input_length=256 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=4 --lr=1e-5 --log_name="Resp_init_4layer_lr1e5_KnowFromKERS_cand20_Input256" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=0 --kers_input_length=256 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=6 --lr=1e-5 --log_name="Resp_init_6layer_lr1e5_KnowFromKERS_cand20_Input256" 

# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=0 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=2 --lr=1e-5 --log_name="Resp_init_2layer_lr1e5_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=0 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=2 --lr=1e-4 --log_name="Resp_init_2layer_lr1e4_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=0 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=2 --lr=1e-3 --log_name="Resp_init_2layer_lr1e3_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=1 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=4 --lr=1e-5 --log_name="Resp_init_4layer_lr1e5_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=1 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=4 --lr=1e-4 --log_name="Resp_init_4layer_lr1e4_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=1 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=4 --lr=1e-3 --log_name="Resp_init_4layer_lr1e3_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=2 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=6 --lr=1e-5 --log_name="Resp_init_6layer_lr1e5_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=2 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=6 --lr=1e-4 --log_name="Resp_init_6layer_lr1e4_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=2 --kers_know_candidate_knowledge_num=20 --kers_resp_layer_num=6 --lr=1e-3 --log_name="Resp_init_6layer_lr1e3_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=3 --kers_know_candidate_knowledge_num=20 --kers_batch_size=8 --kers_resp_layer_num=12 --lr=1e-5 --log_name="Resp_init_12layer_lr1e5_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=3 --kers_know_candidate_knowledge_num=20 --kers_batch_size=8 --kers_resp_layer_num=12 --lr=1e-4 --log_name="Resp_init_12layer_lr1e4_KnowFromKERS_cand20" 
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=3 --kers_know_candidate_knowledge_num=20 --kers_batch_size=8 --kers_resp_layer_num=12 --lr=1e-3 --log_name="Resp_init_12layer_lr1e3_KnowFromKERS_cand20" 

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=7b_len512_promptHJ_3e4 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=test --lora_weights=2023-12-30_150734_7b_len512_promptHJ_3e4_llama_log.txt_Epoch
# CUDA_VISIBLE_DEVICES=1 python llama_main_finetune.py --log_name=7b_len512_231207 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=test --lora_weights=2023-12-07_235521_Llama7b_chat_len512_llama_log.txt_Epoch
# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=7b_len512_promptHJ_3e4 --epoch=10 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=train_test

# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=0 --inputWithKnowledge --kers_know_candidate_knowledge_num=0 --log_name="Resp_KnowpredFromKERS_cand20_LARGE" --kers_generator=facebook/bart-large
# python kers_main.py --version='2' --task=resp --num_epochs=10 --device=1 --inputWithKnowledge --kers_know_candidate_knowledge_num=20 --log_name="Resp_KnowpredFromKERS_candNo_LARGE" --kers_generator=facebook/bart-large

# python kers_main.py --version='2' --task=know_resp --num_epochs=10 --device=0 --inputWithKnowledge --kers_know_candidate_knowledge_num=0 --log_name="Know_NoKnowledge_resp_LARGE" --kers_generator=facebook/bart-large
# python kers_main.py --version='2' --task=know_resp --num_epochs=10 --device=1 --inputWithKnowledge --kers_know_candidate_knowledge_num=20 --log_name="Know_20Knowledge_resp_LARGE" --kers_generator=facebook/bart-large
# python kers_main.py --version='2' --task=know_resp --num_epochs=10 --device=2 --inputWithKnowledge --kers_know_candidate_knowledge_num=0 --log_name="Know_respTask_NoKnowledge" 
# python kers_main.py --version='2' --task=know_resp --num_epochs=10 --device=3 --inputWithKnowledge --kers_know_candidate_knowledge_num=20 --log_name="Know_respTask_20Knowledge" 


# python komain.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --lr=1e-5 --log_name=794_RG2_T2Conf70_PsdBM25_1e5 --model_name=794_RG2_T2Conf70_PsdBM25_1e5 --topk_topic=2 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --pseudo_pos_num=2  --device=0 --pseudo_labeler=bm25  --task_iter=3 --knowledge_method=facebook/mcontriever
# python komain.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --lr=1e-5 --log_name=794_CL1_T2Conf70_PsdBM25_NoIdea_1e5 --model_name=794_CL1_T2Conf70_PsdBM25_NoIdea_1e5 --topk_topic=0 --train_ablation=CL --pseudo_pos_num=1  --device=1 --pseudo_labeler=bm25  --task_iter=3 --knowledge_method=facebook/mcontriever

# python komain.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --lr=1e-6 --log_name=794_RG2_T2Conf70_PsdBM25_1e6 --model_name=794_RG2_T2Conf70_PsdBM25_1e6 --topk_topic=2 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --pseudo_pos_num=2  --device=0 --pseudo_labeler=bm25  --task_iter=3 --knowledge_method=facebook/mcontriever
# python komain.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --lr=1e-6 --log_name=794_CL1_T2Conf70_PsdBM25_NoIdea_1e6 --model_name=794_CL1_T2Conf70_PsdBM25_NoIdea_1e6 --topk_topic=0 --train_ablation=CL --pseudo_pos_num=1  --device=1 --pseudo_labeler=bm25  --task_iter=3 --knowledge_method=facebook/mcontriever

# python komain.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --lr=1e-4 --log_name=794_RG2_T2Conf70_PsdBM25_1e4 --model_name=794_RG2_T2Conf70_PsdBM25_1e4 --topk_topic=2 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --pseudo_pos_num=2  --device=0 --pseudo_labeler=bm25  --task_iter=3 --knowledge_method=facebook/mcontriever
# python komain.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --lr=1e-4 --log_name=794_CL1_T2Conf70_PsdBM25_NoIdea_1e4 --model_name=794_CL1_T2Conf70_PsdBM25_NoIdea_1e4 --topk_topic=0 --train_ablation=CL --pseudo_pos_num=1  --device=1 --pseudo_labeler=bm25  --task_iter=3 --knowledge_method=facebook/mcontriever


## doc 을 3, 5로 변경해봄 \\ rag_context_input에서 doc + dialog만 들어가도록 변경
# python main.py --gpu=0 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=5 --log_name=OnlyDOC5rag_n_docs_RAG_No_DPR --knowledge_method=dpr --rag_our_model=dpr   --rag_model=token --idea=0
# python main.py --gpu=1 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=5 --log_name=OnlyDOC5_RAG_No_CotMAE --knowledge_method=cotmae --rag_our_model=cotmae   --rag_model=token --idea=0
# python main.py --gpu=2 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=5 --log_name=OnlyDOC5_RAG_No_Contriever --knowledge_method=contriever --rag_our_model=contriever   --rag_model=token --idea=0
# python main.py --gpu=3 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=5 --log_name=OnlyDOC5_RAG_OUR_DPR --knowledge_method=dpr --rag_our_model=dpr   --rag_model=token --idea=1_2
# python main.py --gpu=0 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=5 --log_name=OnlyDOC5_RAG_OUR_CotMAE --knowledge_method=cotmae --rag_our_model=cotmae   --rag_model=token --idea=1_2
# python main.py --gpu=1 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=5 --log_name=OnlyDOC5_RAG_OUR_Contriever --knowledge_method=contriever --rag_our_model=contriever   --rag_model=token --idea=1_2

# python main.py --gpu=0 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=3 --log_name=OnlyDOC3_RAG_No_DPR --knowledge_method=dpr --rag_our_model=dpr   --rag_model=token --idea=0
# python main.py --gpu=1 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=3 --log_name=OnlyDOC3_RAG_No_CotMAE --knowledge_method=cotmae --rag_our_model=cotmae   --rag_model=token --idea=0
# python main.py --gpu=2 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=3 --log_name=OnlyDOC3_RAG_No_Contriever --knowledge_method=contriever --rag_our_model=contriever   --rag_model=token --idea=0
# python main.py --gpu=3 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=3 --log_name=OnlyDOC3_RAG_OUR_DPR --knowledge_method=dpr --rag_our_model=dpr   --rag_model=token --idea=1_2
# python main.py --gpu=0 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=3 --log_name=OnlyDOC3_RAG_OUR_CotMAE --knowledge_method=cotmae --rag_our_model=cotmae   --rag_model=token --idea=1_2
# python main.py --gpu=1 --task=resp --task_iter=3 --rag_context_input_only_dialog_doc --rag_epochs=5 --rag_onlyDecoderTune --rag_lr=1e-5 --rag_n_docs=3 --log_name=OnlyDOC3_RAG_OUR_Contriever --knowledge_method=contriever --rag_our_model=contriever   --rag_model=token --idea=1_2

####
# python main.py --gpu=0 --task=resp --log_name=RAG_NoIdea_DPR_RAG --knowledge_method=dpr --rag_our_model=dpr --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=0
# python main.py --gpu=0 --task=resp --log_name=RAG_NoIdea_DPR_RAG --knowledge_method=dpr --rag_our_model=dpr --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=0

# python main.py --gpu=1 --task=resp --log_name=RAG_NoIdea_CotMAE_RAG --knowledge_method=cotmae --rag_our_model=cotmae --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=0
# python main.py --gpu=1 --task=resp --log_name=RAG_NoIdea_CotMAE_RAG --knowledge_method=cotmae --rag_our_model=cotmae --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=0

# python main.py --gpu=2 --task=resp --log_name=RAG_NoIdea_Contriever_RAG --knowledge_method=contriever --rag_our_model=contriever --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=0
# python main.py --gpu=2 --task=resp --log_name=RAG_NoIdea_Contriever_RAG --knowledge_method=contriever --rag_our_model=contriever --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=0

# python main.py --gpu=3 --task=resp --log_name=RAG_OUR_DPR_RAG --knowledge_method=dpr --rag_our_model=dpr --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=1_2
# python main.py --gpu=3 --task=resp --log_name=RAG_OUR_DPR_RAG --knowledge_method=dpr --rag_our_model=dpr --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=1_2

# python main.py --gpu=0 --task=resp --log_name=RAG_OUR_CotMAE_RAG --knowledge_method=cotmae --rag_our_model=cotmae --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=1_2
# python main.py --gpu=0 --task=resp --log_name=RAG_OUR_CotMAE_RAG --knowledge_method=cotmae --rag_our_model=cotmae --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=1_2

# python main.py --gpu=1 --task=resp --log_name=RAG_OUR_Contriever_RAG --knowledge_method=contriever --rag_our_model=contriever --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=1_2
# python main.py --gpu=1 --task=resp --log_name=RAG_OUR_Contriever_RAG --knowledge_method=contriever --rag_our_model=contriever --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=1_2




# python kers_main.py --version='2' --num_epochs=10 --device=0 --inputWithKnowledge --gtpred --log_name="KnowTask_DPredGoal_PseudoKnowTrain" --usePseudoTrain
# python kers_main.py --version='2' --num_epochs=10 --device=1 --inputWithKnowledge --inputWithTopic --gtpred --log_name="KnowTask_DPredGoalTopic_PseudoKnowTrain" --usePseudoTrain

# python komain.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_RG2_T2Conf70_PsdBM25 --model_name=794_RG2_T2Conf70_PsdBM25 --topk_topic=2 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --pseudo_pos_num=2  --device=0 --pseudo_labeler=bm25  --task_iter=3 --knowledge_method=facebook/mcontriever
# python komain.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_CL1_T2Conf70_PsdBM25_NoIdea --model_name=794_CL1_T2Conf70_PsdBM25_NoIdea --topk_topic=0 --train_ablation=CL --pseudo_pos_num=1  --device=1 --pseudo_labeler=bm25  --task_iter=3 --knowledge_method=facebook/mcontriever
# #  --topk_topic=0 --train_ablation=CL --pseudo_pos_num=1


# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_RG2_Rev_T3Conf70_PsdBM25 --model_name=794_RG2_Rev_T3Conf70_PsdBM25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --pseudo_pos_num=2  --device=0 --pseudo_labeler=bm25  --know_iter=3 --train_ablation_reverse
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_RG3_Rev_T3Conf70_PsdBM25 --model_name=794_RG3_Rev_T3Conf70_PsdBM25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --pseudo_pos_num=3  --device=1 --pseudo_labeler=bm25  --know_iter=3 --train_ablation_reverse
####################
# GL, RG, CL 에서 pseudo num 1, 2, 3 에 대한 실험
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_RG1_T3Conf70_PsdBM25 --model_name=794_RG1_T3Conf70_PsdBM25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --pseudo_pos_num=1  --device=0 --pseudo_labeler=bm25  --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_RG2_T3Conf70_PsdBM25 --model_name=794_RG2_T3Conf70_PsdBM25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --pseudo_pos_num=2  --device=1 --pseudo_labeler=bm25  --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_GL1_T3Conf70_PsdBM25 --model_name=794_GL1_T3Conf70_PsdBM25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=GL --pseudo_pos_num=1  --device=0 --pseudo_labeler=bm25  --know_iter=3
####################

# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_CL1_T3Conf70_PsdBM25 --model_name=794_CL1_T3Conf70_PsdBM25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=CL --pseudo_pos_num=1  --device=0 --pseudo_labeler=bm25  --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_CL2_T3Conf70_PsdBM25 --model_name=794_CL2_T3Conf70_PsdBM25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=CL --pseudo_pos_num=2  --device=0 --pseudo_labeler=bm25  --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_CL3_T3Conf70_PsdBM25 --model_name=794_CL3_T3Conf70_PsdBM25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=CL --pseudo_pos_num=3  --device=1 --pseudo_labeler=bm25  --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_RG3_T3Conf70_PsdBM25 --model_name=794_RG3_T3Conf70_PsdBM25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --pseudo_pos_num=3  --device=1 --pseudo_labeler=bm25  --know_iter=3


# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_TH_GL3_T3Conf70_Psd_bm25 --model_name=794_TH_GL3_T3Conf70_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=GL --pseudo_pos_num=3  --device=0 --pseudo_labeler=bm25  --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794_TH_GL2_T3Conf70_Psd_bm25 --model_name=794_TH_GL2_T3Conf70_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=GL --pseudo_pos_num=2  --device=1 --pseudo_labeler=bm25  --know_iter=3
####################


# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T2Conf100_Psd_bm25 --model_name=794RG_T2Conf100_Psd_bm25 --topk_topic=2 --know_item_select=conf --topic_conf=1.0 --train_ablation=RG   --device=1 --pseudo_labeler=bm25  --know_iter=3

# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf0_Psd_bm25 --model_name=794RG_T3Conf0_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.0 --train_ablation=RG   --device=0 --pseudo_labeler=bm25  --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf10_Psd_bm25 --model_name=794RG_T3Conf10_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.1 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf20_Psd_bm25 --model_name=794RG_T3Conf20_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.2 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf30_Psd_bm25 --model_name=794RG_T3Conf30_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.3 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf40_Psd_bm25 --model_name=794RG_T3Conf40_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.4 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf50_Psd_bm25 --model_name=794RG_T3Conf50_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.5 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf60_Psd_bm25 --model_name=794RG_T3Conf60_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.6 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf70_Psd_bm25 --model_name=794RG_T3Conf70_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf80_Psd_bm25 --model_name=794RG_T3Conf80_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf90_Psd_bm25 --model_name=794RG_T3Conf90_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.9 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf100_Psd_bm25 --model_name=794RG_T3Conf100_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=1.0 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3

# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf0_Psd_bm25 --model_name=794RG_T4Conf0_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.0 --train_ablation=RG   --device=0 --pseudo_labeler=bm25  --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf10_Psd_bm25 --model_name=794RG_T4Conf10_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.1 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf20_Psd_bm25 --model_name=794RG_T4Conf20_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.2 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf30_Psd_bm25 --model_name=794RG_T4Conf30_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.3 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf40_Psd_bm25 --model_name=794RG_T4Conf40_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.4 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf50_Psd_bm25 --model_name=794RG_T4Conf50_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.5 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf60_Psd_bm25 --model_name=794RG_T4Conf60_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.6 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf70_Psd_bm25 --model_name=794RG_T4Conf70_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf80_Psd_bm25 --model_name=794RG_T4Conf80_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf90_Psd_bm25 --model_name=794RG_T4Conf90_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=0.9 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --know_iter=4
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T4Conf100_Psd_bm25 --model_name=794RG_T4Conf100_Psd_bm25 --topk_topic=4 --know_item_select=conf --topic_conf=1.0 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=4


####################
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf60_Psd_bm25 --model_name=794RG_T3Conf60_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.6 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf40_Psd_bm25 --model_name=794RG_T3Conf40_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.4 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf30_Psd_bm25 --model_name=794RG_T3Conf30_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.3 --train_ablation=RG --device=0 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf0_Psd_bm25 --model_name=794RG_T3Conf0_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.0 --train_ablation=RG   --device=1 --pseudo_labeler=bm25  --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf10_Psd_bm25 --model_name=794RG_T3Conf10_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.1 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf20_Psd_bm25 --model_name=794RG_T3Conf20_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.2 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf30_Psd_bm25 --model_name=794RG_T3Conf30_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.3 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf40_Psd_bm25 --model_name=794RG_T3Conf40_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.4 --train_ablation=RG --device=2 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf50_Psd_bm25 --model_name=794RG_T3Conf50_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.5 --train_ablation=RG --device=2 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf60_Psd_bm25 --model_name=794RG_T3Conf60_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.6 --train_ablation=RG --device=2 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf70_Psd_bm25 --model_name=794RG_T3Conf70_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.7 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf80_Psd_bm25 --model_name=794RG_T3Conf80_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf90_Psd_bm25 --model_name=794RG_T3Conf90_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=0.9 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3
# python main.py --task=know_pred_k --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T3Conf100_Psd_bm25 --model_name=794RG_T3Conf100_Psd_bm25 --topk_topic=3 --know_item_select=conf --topic_conf=1.0 --train_ablation=RG --device=3 --pseudo_labeler=bm25 --know_iter=3


# python kers_main.py --task=resp --version=2 --device=0 --kers_candidate_knowledge_num=3 --log_name=KERS_BASE_PsdShuffledTop3 --kers_generator=facebook/bart-base
# python kers_main.py --task=resp --version=2 --device=1 --kers_candidate_knowledge_num=3 --log_name=KERS_LARGE_PsdShuffledTop3 --kers_generator=facebook/bart-large --kers_batch_size=8

# python kers_main.py --task=resp --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Ret3_1e-4" --kers_candidate_knowledge_num=3 --lr=1e-4 --gpu=1 --bert_name='skt/kobert-base-v1' --kers_batch_size=8 --kers_pretrain_epochs=0

# python kers_main.py --task=resp --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Ret1_aug_1e-5" --kers_candidate_knowledge_num=1 --lr=1e-5 --gpu=1 --bert_name='skt/kobert-base-v1' --kers_batch_size=8
# python kers_main.py --task=resp --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Ret3_aug_1e-5" --kers_candidate_knowledge_num=3 --lr=1e-5 --gpu=1 --bert_name='skt/kobert-base-v1' --kers_batch_size=8
# python kers_main.py --task=resp --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Ret5_aug_1e-5" --kers_candidate_knowledge_num=5 --lr=1e-5 --gpu=1 --bert_name='skt/kobert-base-v1' --kers_batch_size=8
# python kers_main.py --task=resp --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Ret5_aug_3e-4" --kers_candidate_knowledge_num=5 --lr=3e-4 --gpu=0 --bert_name='skt/kobert-base-v1' --kers_batch_size=8
# python kers_main.py --task=resp --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Ret5_aug_1e-4" --kers_candidate_knowledge_num=5 --lr=1e-4 --gpu=0 --bert_name='skt/kobert-base-v1' --kers_batch_size=8
# python kers_main.py --task=resp --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Ret5_aug_1e-6" --kers_candidate_knowledge_num=5 --lr=1e-6 --gpu=0 --bert_name='skt/kobert-base-v1' --kers_batch_size=8





# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=7b_len512_promptHJ_3e4 --epoch=10 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=train_test

# ---------------------- 20231209 ----------------#
# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=7b_len512_promptHJ_2e4 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=test --lora_weights=2023-12-08_145254_7b_len512_promptHJ_2e4_llama_log.txt_Epoch

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=7b_len512_promptHJ_2e4 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=test --lora_weights=2023-12-08_145254_7b_len512_promptHJ_2e4_llama_log.txt_Epoch6

# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=7b_len512_promptHJ_3e4 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=train_test
# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=7b_len512_promptHJ_1e4 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --learning_rate=1e-4
# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=7b_len512_promptHJ_1e5 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --learning_rate=1e-5 --mode=train_test --batch_size=8 # 터짐
# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=7b_len512_promptHJ_2e4 --epoch=7 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --learning_rate=2e-4 


# CUDA_VISIBLE_DEVICES=0 python llama_main_finetune.py --log_name=Llama7b_chat_len512_2023-12-07_235521_len512_Epoch5_Test --epoch=5 --base_model=meta-llama/Llama-2-7b-chat-hf --llama_input_maxlen=512 --mode=test --lora_weights=2023-12-07_235521_Llama7b_chat_len512_llama_log.txt_Epoch5


# python llama_main_finetune.py --log_name=Llama7b_chat --epoch=5 --base_model=meta-llama/Llama-2-7b-chat-hf

# python main.py --gpu=0 --task=resp --log_name=DPROUR_RAG_OnlyDoc_Dialog --knowledge_method=dpr --rag_our_model=dpr --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=0_1
# python main.py --gpu=0 --task=resp --log_name=CotMAE_RAG_OnlyDoc_Dialog --knowledge_method=cotmae --rag_our_model=cotmae --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=0_1
# python main.py --gpu=0 --task=resp --log_name=Contriever_RAG_OnlyDoc_Dialog --knowledge_method=contriever --rag_our_model=contriever --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token --idea=0_1

# python main.py --gpu=0 --task=resp --rag_lr=1e-5 --rag_epochs=13 --rag_onlyDecoderTune --log_name=DPROUR_RAG_OnlyDocDialog_Top1 --knowledge_method=dpr --rag_our_model=dpr --rag_model=token --idea=top1

# python main.py --gpu=0 --task=resp --rag_lr=1e-5 --rag_epochs=13 --rag_onlyDecoderTune --log_name=DPR_OnlyDocDialog_NoIdea_RAG_TOKEN --knowledge_method=dpr --rag_our_model=dpr --rag_model=token --idea=0
# ---------------------- 20231204 ----------------#
# python main.py --gpu=0 --task=resp --rag_lr=1e-5 --rag_epochs=13 --rag_onlyDecoderTune --log_name=DPR_Top1_RAG_TOKEN --knowledge_method=dpr --rag_our_model=dpr --rag_model=token --idea=top1

# python main.py --gpu=0 --task=resp --rag_lr=1e-5 --rag_epochs=13 --rag_onlyDecoderTune --log_name=DPR_NoIdea_RAG_TOKEN --knowledge_method=dpr --rag_our_model=dpr --rag_model=token --idea=0
# python main.py --gpu=0 --task=resp --rag_lr=1e-5 --rag_epochs=13 --rag_onlyDecoderTune --log_name=Contriever_NoIdea_RAG_TOKEN --knowledge_method=contriever --rag_our_model=contriever --rag_model=token --idea=0
# python main.py --gpu=1 --task=resp --rag_lr=1e-5 --rag_epochs=13 --rag_onlyDecoderTune --log_name=CotMAE_NoIdea_RAG_TOKEN --knowledge_method=cotmae --rag_our_model=cotmae --rag_model=token --idea=0

# python kers_main.py --task=resp --version=2 --kers_batch_size=10 --kers_generator=facebook/bart-large --device=3 --kers_candidate_knowledge_num=5  --log_name=KERS_1e4_PsdShuffledTop5 --lr=1e-4 
# python kers_main.py --task=resp --version=2 --kers_batch_size=10 --kers_generator=facebook/bart-large --device=2 --kers_candidate_knowledge_num=5  --log_name=KERS_1e6_PsdShuffledTop5 --lr=1e-6
# python kers_main.py --task=resp --version=2 --kers_batch_size=10 --kers_generator=facebook/bart-large --device=1 --kers_candidate_knowledge_num=5  --log_name=KERS_5e5_PsdShuffledTop5 --lr=5e-5


# ---------------------- 20231130 ----------------#
# python kers_main.py --task=resp --version=2 --kers_batch_size=10 --kers_generator=facebook/bart-large --device=0 --kers_candidate_knowledge_num=1  --log_name=KERS_PsdShuffledTop1
# python kers_main.py --task=resp --version=2 --kers_batch_size=10 --kers_generator=facebook/bart-large --device=1 --kers_candidate_knowledge_num=2  --log_name=KERS_PsdShuffledTop2
# python kers_main.py --task=resp --version=2 --kers_batch_size=10 --kers_generator=facebook/bart-large --device=2 --kers_candidate_knowledge_num=5  --log_name=KERS_PsdShuffledTop5
# python kers_main.py --task=resp --version=2 --kers_batch_size=10 --kers_generator=facebook/bart-large --device=3 --kers_candidate_knowledge_num=10 --log_name=KERS_PsdShuffledTop10
# python kers_main.py --task=resp --version=2 --kers_batch_size=10 --kers_generator=facebook/bart-large --device=1 --kers_candidate_knowledge_num=20 --log_name=KERS_PsdShuffledTop20
# python kers_main.py --task=resp --version=2 --kers_batch_size=10 --kers_generator=facebook/bart-large --device=0 --kers_candidate_knowledge_num=2 --log_name=KERS_PsdTop2 --kers_candidate_knowledge_shuffle=''

# ---------------------- 20231130 ----------------#
# python kers_main.py --task=resp --version=2 --device=0 --kers_candidate_knowledge_num=5  --log_name=KERS_PT5_PsdShuffledTop5 --kers_pretrain_epochs=5
# python kers_main.py --task=resp --version=2 --device=1 --kers_candidate_knowledge_num=10 --log_name=KERS_PT5_PsdShuffledTop10 --kers_pretrain_epochs=5
# python kers_main.py --task=resp --version=2 --device=2 --kers_candidate_knowledge_num=15 --log_name=KERS_PT5_PsdShuffledTop15 --kers_pretrain_epochs=5
# python kers_main.py --task=resp --version=2 --device=3 --kers_candidate_knowledge_num=15 --log_name=KERS_PT5_PsdShuffledTop15 --kers_pretrain_epochs=5



# python main.py --gpu=1 --task=resp --rag_lr=1e-5 --rag_epochs=13 --rag_onlyDecoderTune --log_name=DPROUR_RAG_SEQ --knowledge_method=dpr --rag_our_model=dpr --rag_model=sequence
# python main.py --gpu=1 --task=resp --rag_lr=1e-5 --rag_epochs=13 --rag_onlyDecoderTune --log_name=Contriever_RAG_SEQ --knowledge_method=contriever --rag_our_model=contriever --rag_model=sequence
# python main.py --gpu=1 --task=resp --rag_lr=1e-5 --rag_epochs=13 --rag_onlyDecoderTune --log_name=CotMAE_RAG_SEQ --knowledge_method=cotmae --rag_our_model=cotmae --rag_model=sequence

# ---------------------- 20231121 ----------------#

# python kers_main.py --task=resp --version=2 --device=0 --kers_candidate_knowledge_num=20 --log_name=KERS_PsdShuffledTop20
# python kers_main.py --task=resp --version=2 --device=0 --kers_candidate_knowledge_num=10 --log_name=KERS_PsdShuffledTop10
# python kers_main.py --task=resp --version=2 --device=0 --kers_candidate_knowledge_num=5 --log_name=KERS_PsdShuffledTop5
# python kers_main.py --task=resp --version=2 --gpu=0  --log_name=PreTrain_KERS_with_20ShuffledKnowledge --do_pretrain 

# python main.py --gpu=1 --task=resp --log_name=DPROUR_RAG --knowledge_method=dpr --rag_our_model=dpr --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token
# python main.py --gpu=1 --task=resp --log_name=Contriever_RAG --knowledge_method=contriever --rag_our_model=contriever --rag_lr=1e-5 --rag_epochs=15 --rag_onlyDecoderTune --rag_model=token

# python main.py --task=pred_k --model_name=ContrieverDPR_CL_Psd_BM25 --log_name=ContrieverDPR_CL_Psd_BM25_contriever로NoIdea학습 --topk_topic=0 --train_ablation=CL --pseudo_pos_num=1 --topk_topic=0 --contriever
# python main.py --task=pred_k --model_name=ContrieverOUR_GL_Psd_BM25 --log_name=ContrieverOUR_GL_Psd_BM25_contriever로OurIdea학습 --topk_topic=2 --train_ablation=RG --pseudo_pos_num=2 --topk_topic=2 --contriever
# python main.py --task=pred_k --model_name=CotMAE_CL_Psd_BM25 --log_name=CotMAE_CL_Psd_BM25_coamae로NoIdea학습 --topk_topic=0 --train_ablation=CL --pseudo_pos_num=1 --topk_topic=0 --cotmae
# python main.py --task=pred_k --model_name=CotMAEOUR_GL2_Psd_BM25 --log_name=CotMAEOUR_GL2_Psd_BM25__cotmae로OurIdea학습 --topk_topic=2 --train_ablation=RG --pseudo_pos_num=2 --topk_topic=2 --cotmae
# python main.py --task=pred_k --model_name=DPRR_CL_Psd_BM25 --log_name=_DPRR_CL_Psd_BM25__DPR로NoIdea학습 --topk_topic=0 --train_ablation=CL --pseudo_pos_num=1 --topk_topic=0 
# python main.py --task=pred_k --model_name=RB_794RG_topic2_conf70_hj --log_name=RB_794RG_topic2_conf70_hj__DPR로OurIdea학습 --topk_topic=2 --train_ablation=RG --pseudo_pos_num=2 --topk_topic=2 



# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=CotMAEOUR_GL2_Psd_BM25 --model_name=CotMAEOUR_GL2_Psd_BM25 --topk_topic=2 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --pseudo_pos_num=2 --cotmae # Contriever로 initialize하고, 우리 idea 2개 다 적용
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=CotMAE_CL_Psd_BM25 --model_name=CotMAE_CL_Psd_BM25 --topk_topic=0 --train_ablation=CL --device=0 --pseudo_labeler=bm25 --pseudo_pos_num=1 --cotmae # CotMAE에 DPR 방식 학습 (Top1이용, negative sampling)

# ---------------------- 20231121 ----------------#

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=ContrieverOUR_GL_Psd_BM25 --model_name=ContrieverOUR_GL_Psd_BM25 --topk_topic=2 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --pseudo_pos_num=2 --contriever # Contriever로 initialize하고, 우리 idea 2개 다 적용
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=DPRR_CL_Psd_BM25 --model_name=DPRR_CL_Psd_BM25 --topk_topic=0 --train_ablation=CL --device=1 --pseudo_labeler=bm25 --pseudo_pos_num=1 # DPR 방식 학습 (Top1이용, negative sampling)
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=ContrieverDPR_CL_Psd_BM25 --model_name=ContrieverDPR_CL_Psd_BM25 --topk_topic=0 --train_ablation=CL --device=0 --pseudo_labeler=bm25 --pseudo_pos_num=1 --contriever # DPR 방식 학습 (Top1이용, negative sampling)

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=OUR_RG_T2Conf80_Psd_BM25 --model_name=OUR_RG_T2Conf80_Psd_BM25 --topk_topic=2 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --contriever --pseudo_pos_num=1

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=Contriever_DPR_794RG_T2Conf80_Psd_BM25 --model_name=Contriever_DPR_794RG_T2Conf80_Psd_BM25 --topk_topic=2 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=1 --pseudo_labeler=bm25 --contriever --pseudo_pos_num=1


# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T2Conf80_Psd_CotMAE --model_name=794RG_T2Conf80_Psd_CotMAE --topk_topic=2 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=0 --pseudo_labeler=cotmae
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T2Conf80_Psd_Contriever --model_name=794RG_T2Conf80_Psd_Contriever --topk_topic=2 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=0 --pseudo_labeler=contriever
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_T2Conf80_Psd_DPR --model_name=794RG_T2Conf80_Psd_DPR --topk_topic=2 --know_item_select=conf --topic_conf=0.8 --train_ablation=RG --device=1 --pseudo_labeler=dpr

# ---------------------- 20231120 ----------------#
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=cot --log_name="cotmae_base_uncased"
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=dpr --log_name=PreTrainedDPR 
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=contriever  --log_name=contriever-msmarco 

# python pseudo_labeler.py --mode=test --how=resp_uttr_item --gpu=1 --save --score_method=cot --log_name="cotmae_base_msmarco_reranker"
# python pseudo_labeler.py --mode=test --how=resp_uttr_item --gpu=1 --save --score_method=cot --log_name="cotmae_base_msmarco_retriever"

# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=bm25  --log_name=BM25
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=contriever  --log_name=contriever-msmarco 
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=0 --save --score_method=dpr --log_name=PreTrainedDPR 
# python pseudo_labeler.py --mode=train_dev_test --how=resp_uttr_item --gpu=1 --save --score_method=cot --log_name=Cotmae 

# python preprocess_bm25.py --mode=test --how=resp_uttr_item --score_method=contriever --gpu=1

#----------------- 20231103 ------------------#
# python preprocess_bm25.py --mode=test --how=resp
# python preprocess_bm25.py --mode=test --how=resp_uttr
# python preprocess_bm25.py --mode=test --how=resp_item
# python preprocess_bm25.py --mode=test --how=resp_uttr_item 
# python preprocess_bm25.py --mode=test --how=uttr
# python preprocess_bm25.py --mode=test --how=uttr_item
# python preprocess_bm25.py --mode=test --how=item


#----------------- 20231101 ------------------#
# python llama_main.py --gpu=1 --base_model=meta-llama/Llama-2-13b-chat-hf --log_name=Llama13B_2
# python llama_main.py --gpu=0 --base_model=meta-llama/Llama-2-7b-chat-hf --log_name=Llama7B

# python lm_main.py --fast --version=2 --gpu=1 --uni_epochs=7 --uni_model_name='google/flan-t5-large' --uni_batch_size=8 --log_name="T5-large_FineTune" --finetune
# python lm_main.py --fast --version=2 --gpu=1 --uni_epochs=2 --uni_model_name='google/flan-t5-xl' --uni_batch_size=1 --log_name="T5-xl" 
# python lm_main.py --fast --version=2 --gpu=0 --uni_epochs=2 --uni_model_name='google/flan-t5-xxl' --uni_batch_size=1 --log_name="T5-xxl_13b" 

# python lm_main.py --fast --version=2 --gpu=0 --uni_epochs=2 --uni_model_name='google/flan-t5-large' --uni_batch_size=1 --log_name="T5-large" 
# ["--gpu=1","--fast", "--topic_rq=none", "--log_name=DEBUG", "--uni_model_name=google/flan-t5-xxl", "--uni_batch_size=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794RG_topic_Top0_th --model_name=794RG_topic_Top0_th --topk_topic=0 --know_item_select=conf --train_ablation=RG --device=1
# python unimind_main.py --fast --version=2 --gpu=1 --method=t5 --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='google/flan-t5-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=8 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="T5_794_T5_RecGen_Cum2_Conf80_1e5" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=t5 --uni_lr=1e-6 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='google/flan-t5-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=8 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="T5_794_T5_RecGen_Cum2_Conf80_1e6" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=t5 --uni_lr=5e-4 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='google/flan-t5-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=8 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="T5_794_T5_RecGen_Cum2_Conf80_5e4" 

#----------------- 20231017 ------------------#
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic_Top1_th --model_name=794CL_topic_Top1_th --topk_topic=1 --know_item_select=top --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic_Top1_th --model_name=794CL_topic_Top1_th --topk_topic=1 --know_item_select=top --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic_Top2_th --model_name=794CL_topic_Top2_th --topk_topic=2 --know_item_select=top --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic_Top2_th --model_name=794CL_topic_Top2_th --topk_topic=2 --know_item_select=top --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf90_th --model_name=794CL_topic2_conf90_th --topk_topic=2 --topic_conf=0.9 --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf90_th --model_name=794CL_topic2_conf90_th --topk_topic=2 --topic_conf=0.9 --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf80_th --model_name=794CL_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf80_th --model_name=794CL_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf70_th --model_name=794CL_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf70_th --model_name=794CL_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf60_th --model_name=794CL_topic2_conf60_th --topk_topic=2 --topic_conf=0.6 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf60_th --model_name=794CL_topic2_conf60_th --topk_topic=2 --topic_conf=0.6 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf50_th --model_name=794CL_topic2_conf50_th --topk_topic=2 --topic_conf=0.5 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=794CL_topic2_conf50_th --model_name=794CL_topic2_conf50_th --topk_topic=2 --topic_conf=0.5 --train_ablation=CL --device=1

#----------------- 20231016 ------------------#
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.5 --log_name="RESP_794Uni_RECG_Cum2_Conf50" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.6 --log_name="RESP_794Uni_RECG_Cum2_Conf60" 
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.7 --log_name="RESP_794Uni_RECG_Cum2_Conf70" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="RESP_794Uni_RECG_Cum2_Conf80" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.9 --log_name="RESP_794Uni_RECG_Cum2_Conf90" 
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=1 --log_name="RESP_794Uni_RECG_Cum2_Conf100" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=1 --topic_rq=top --log_name="RESP_794Uni_RECG_Top1" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --topic_rq_label=resp --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=top --log_name="RESP_794Uni_RECG_Top2" 

# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.5 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf50" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.6 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf60" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf80" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.9 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf90" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.7 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf70" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=1 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf100" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=1 --topic_rq=top --log_name="1e-5_794Uni_RECGEN_Top1" 
# python unimind_main.py --fast --version=2 --gpu=0 --method=unimind --uni_lr=1e-5 --uni_epochs=7 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=top --log_name="1e-5_794Uni_RECGEN_Top2" 

# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/pkl_aaai/test_pred_aug_dataset.pkl')))
# 0.6848329048843188
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/pkl_768/test_pred_aug_dataset.pkl')))
# 0.6699228791773779
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/pkl_794/test_pred_aug_dataset.pkl')))
# 0.6930591259640103
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/gt_test_pred_aug_dataset.pkl')))
# 0.6699228791773779
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/gt_test_pred_aug_dataset0.pkl')))
# 0.6822622107969152
# pred_aug_topic_hi1(utils.read_pkl(os.path.join(home, 'data/2/pred_aug/gt_test_pred_aug_dataset1.pkl')))
# 0.6910025706940874

#----------------- 20231013 ------------------#
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_th --model_name=RB_794RG_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_th --model_name=RB_794RG_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_th --model_name=RB_794RG_topic2_conf80_th --topk_topic=2 --topic_conf=0.8 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_th --model_name=RB_794RG_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_th --model_name=RB_794RG_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_th --model_name=RB_794RG_topic2_conf70_th --topk_topic=2 --topic_conf=0.7 --train_ablation=CL --device=1

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top1topic_hj --model_name=RB_794RG_Top1topic_hj --topk_topic=1 --know_item_select=top --train_ablation=RG --device=2

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_794RG_Top2topic_hj --model_name=RB_794RG_Top2topic_hj --topk_topic=2 --know_item_select=top --train_ablation=RG --device=2

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf100_hj --model_name=RB_794RG_topic2_conf100_hj --topk_topic=2 --topic_conf=1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf90_hj --model_name=RB_794RG_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_hj --model_name=RB_794RG_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_hj --model_name=RB_794RG_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf60_hj --model_name=RB_794RG_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf50_hj --model_name=RB_794RG_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf40_hj --model_name=RB_794RG_topic2_conf40_hj --topk_topic=2 --topic_conf=0.4 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf30_hj --model_name=RB_794RG_topic2_conf30_hj --topk_topic=2 --topic_conf=0.3 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf20_hj --model_name=RB_794RG_topic2_conf20_hj --topk_topic=2 --topic_conf=0.2 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf10_hj --model_name=RB_794RG_topic2_conf10_hj --topk_topic=2 --topic_conf=0.1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf100_hj --model_name=RB_794RG_topic2_conf100_hj --topk_topic=2 --topic_conf=1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf90_hj --model_name=RB_794RG_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_hj --model_name=RB_794RG_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_hj --model_name=RB_794RG_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf60_hj --model_name=RB_794RG_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf50_hj --model_name=RB_794RG_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf40_hj --model_name=RB_794RG_topic2_conf40_hj --topk_topic=2 --topic_conf=0.4 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf30_hj --model_name=RB_794RG_topic2_conf30_hj --topk_topic=2 --topic_conf=0.3 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf20_hj --model_name=RB_794RG_topic2_conf20_hj --topk_topic=2 --topic_conf=0.2 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf10_hj --model_name=RB_794RG_topic2_conf10_hj --topk_topic=2 --topic_conf=0.1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf100_hj --model_name=RB_794RG_topic2_conf100_hj --topk_topic=2 --topic_conf=1 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf90_hj --model_name=RB_794RG_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf80_hj --model_name=RB_794RG_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf70_hj --model_name=RB_794RG_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf60_hj --model_name=RB_794RG_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf50_hj --model_name=RB_794RG_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf40_hj --model_name=RB_794RG_topic2_conf40_hj --topk_topic=2 --topic_conf=0.4 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf30_hj --model_name=RB_794RG_topic2_conf30_hj --topk_topic=2 --topic_conf=0.3 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf20_hj --model_name=RB_794RG_topic2_conf20_hj --topk_topic=2 --topic_conf=0.2 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf10_hj --model_name=RB_794RG_topic2_conf10_hj --topk_topic=2 --topic_conf=0.1 --train_ablation=RG --device=1

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_RG_794topic2_conf1_hj --model_name=RB_794RG_topic2_conf1_hj --topk_topic=2 --topic_conf=0.01 --train_ablation=RG --device=0





# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=1 --topic_rq=top --log_name="1e-5_794Uni_RECGEN_Top1" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=top --log_name="1e-5_794Uni_RECGEN_Top2" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=1 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf100" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.9 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf90" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf80" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.7 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf70" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.6 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf60" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_score=794 --topk_topic=2 --topic_rq=conf --topic_conf=0.5 --log_name="1e-5_794Uni_RECGEN_Cum2_Conf50" 



# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=1 --log_name="768Uni_RECGEN_Cum2_Conf100" 
# python unimind_main.py --fast --version=2 --gpu=2 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.9 --log_name="768Uni_RECGEN_Cum2_Conf90" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.8 --log_name="768Uni_RECGEN_Cum2_Conf80" 
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.7 --log_name="768Uni_RECGEN_Cum2_Conf70" 
# python unimind_main.py --fast --version=2 --gpu=2 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.6 --log_name="768Uni_RECGEN_Cum2_Conf60" 
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=conf --topic_conf=0.5 --log_name="768Uni_RECGEN_Cum2_Conf50" 
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=1 --topic_rq=top --log_name="768Uni_RECGEN_Top1" 
# python unimind_main.py --fast --version=2 --gpu=2 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --topic_rq=top --log_name="768Uni_RECGEN_Top2" 

# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=1 --topic_rq=top --log_name="768Uni_RECGEN_Top1_기존_onlyresp" 


# python main.py --task=topic --num_epochs=25 --log_name=Topic512_Train_1e-4 --gt_batch_size=32 --gt_max_length=512 --lr=1e-4 --device=0
# python main.py --task=topic --num_epochs=25 --log_name=Topic512_Train_1e-6 --gt_batch_size=32 --gt_max_length=512 --lr=1e-6 --device=1

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RG_768topic_Top1 --model_name=RG_768topic_Top1 --topk_topic=1 --train_ablation=RG --know_item_select=top --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RG_768topic_Top2 --model_name=RG_768topic_Top2 --topk_topic=2 --train_ablation=RG --know_item_select=top --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RG_768topic_Top3 --model_name=RG_768topic_Top3 --topk_topic=3 --train_ablation=RG --know_item_select=top --device=3

#-------------- before 230926 --------------#
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf90_hj --model_name=RB_768GCL2_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf80_hj --model_name=RB_768GCL2_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf70_hj --model_name=RB_768GCL2_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf60_hj --model_name=RB_768GCL2_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic2_conf50_hj --model_name=RB_768GCL2_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=2

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf90_hj --model_name=RB_768GCL2_topic3_conf90_hj --topk_topic=3 --topic_conf=0.9 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf80_hj --model_name=RB_768GCL2_topic3_conf80_hj --topk_topic=3 --topic_conf=0.8 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf70_hj --model_name=RB_768GCL2_topic3_conf70_hj --topk_topic=3 --topic_conf=0.7 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf60_hj --model_name=RB_768GCL2_topic3_conf60_hj --topk_topic=3 --topic_conf=0.6 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic3_conf50_hj --model_name=RB_768GCL2_topic3_conf50_hj --topk_topic=3 --topic_conf=0.5 --train_ablation=RG --device=3

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf90_hj --model_name=RB_768GCL2_topic1_conf90_hj --topk_topic=1 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf80_hj --model_name=RB_768GCL2_topic1_conf80_hj --topk_topic=1 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf70_hj --model_name=RB_768GCL2_topic1_conf70_hj --topk_topic=1 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf60_hj --model_name=RB_768GCL2_topic1_conf60_hj --topk_topic=1 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_768topic1_conf50_hj --model_name=RB_768GCL2_topic1_conf50_hj --topk_topic=1 --topic_conf=0.5 --train_ablation=RG --device=1

#-------------- before 230924 --------------#
# python main.py --task=topic --num_epochs=25 --log_name=Topic512_Train  --device=0 --gt_batch_size=32 --gt_max_length=512
# python main.py --task=goal_topic --num_epochs=25 --log_name=Goal_usepred_Train  --device=1
# python main.py --task=goal_topic --num_epochs=25 --log_name=Goal_Train  --device=3
# python main.py --task=goal_topic --num_epochs=25 --log_name=Goal_Train_WithPooler  --device=2

# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf90_hj --model_name=RB_GCL2_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf80_hj --model_name=RB_GCL2_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf70_hj --model_name=RB_GCL2_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=2
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf60_hj --model_name=RB_GCL2_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=3
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=RB_GCL2_topic2_conf50_hj --model_name=RB_GCL2_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=0
#-------------- before 230923 --------------#
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.9 --log_name="Uni_RECGEN_Cum3_Conf90" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.8 --log_name="Uni_RECGEN_Cum3_Conf80" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.7 --log_name="Uni_RECGEN_Cum3_Conf70" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.6 --log_name="Uni_RECGEN_Cum3_Conf60" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --topic_conf=0.5 --log_name="Uni_RECGEN_Cum3_Conf50" #  BART-Large RQ


# python main.py --task=topic --num_epochs=25 --log_name=Topic_onlyProfileDialog --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf90_hj --model_name=K_GCL1_topic1_conf90_hj --topk_topic=1 --topic_conf=0.9 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf80_hj --model_name=K_GCL1_topic1_conf80_hj --topk_topic=1 --topic_conf=0.8 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf70_hj --model_name=K_GCL1_topic1_conf70_hj --topk_topic=1 --topic_conf=0.7 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf60_hj --model_name=K_GCL1_topic1_conf60_hj --topk_topic=1 --topic_conf=0.6 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf50_hj --model_name=K_GCL1_topic1_conf50_hj --topk_topic=1 --topic_conf=0.5 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL1_topic1_conf40_hj --model_name=K_GCL1_topic1_conf40_hj --topk_topic=1 --topic_conf=0.4 --train_ablation=RG --device=0

#------------- before 230922 --------------#
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=1 --log_name="Uni_RECGEN_Top1" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=2 --log_name="Uni_RECGEN_Top2" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topk_topic=3 --log_name="Uni_RECGEN_Top3" #  BART-Large RQ

# python main.py --task=topic --num_epochs=25 --log_name=Topic_with_userprofile_0 --device=0
# # python main.py --task=topic --num_epochs=25 --log_name=Topic_with_userprofile --device=1
# python main.py --task=topic --num_epochs=25 --log_name=Topic_with_userprofile_0 --device=0
# ----------- before 230920 ---------------#
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf90_hj --model_name=GCL2_topic2_conf90_hj --topk_topic=2 --topic_conf=0.9 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf80_hj --model_name=GCL2_topic2_conf80_hj --topk_topic=2 --topic_conf=0.8 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf70_hj --model_name=GCL2_topic2_conf70_hj --topk_topic=2 --topic_conf=0.7 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf60_hj --model_name=GCL2_topic2_conf60_hj --topk_topic=2 --topic_conf=0.6 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf50_hj --model_name=GCL2_topic2_conf50_hj --topk_topic=2 --topic_conf=0.5 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic2_conf40_hj --model_name=GCL2_topic2_conf40_hj --topk_topic=2 --topic_conf=0.4 --train_ablation=RG --device=0
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf90_hj --model_name=GCL2_topic3_conf90_hj --topk_topic=3 --topic_conf=0.9 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf80_hj --model_name=GCL2_topic3_conf80_hj --topk_topic=3 --topic_conf=0.8 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf70_hj --model_name=GCL2_topic3_conf70_hj --topk_topic=3 --topic_conf=0.7 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf60_hj --model_name=GCL2_topic3_conf60_hj --topk_topic=3 --topic_conf=0.6 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf50_hj --model_name=GCL2_topic3_conf50_hj --topk_topic=3 --topic_conf=0.5 --train_ablation=RG --device=1
# python main.py --task=know --batch_size=32 --know_max_length=128 --num_epochs=20 --input_prompt=dialog_topic --log_name=K_GCL2_topic3_conf40_hj --model_name=GCL2_topic3_conf40_hj --topk_topic=3 --topic_conf=0.4 --train_ablation=RG --device=1

#-------------before 230919-------------------------#
# python unimind_main.py --fast --version=2 --gpu=1 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_conf=0.7 --topk_topic=1 --log_name="Uni_RECGEN_Top1_conf07" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=2 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_conf=0.7 --topk_topic=2 --log_name="Uni_RECGEN_Top2_conf07" #  BART-Large RQ
# python unimind_main.py --fast --version=2 --gpu=3 --method=unimind --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=256 --uni_max_target_length=128 --uni_batch_size=16 --topic_conf=0.7 --topk_topic=3 --log_name="Uni_RECGEN_Top3_conf07" #  BART-Large RQ

# python komain.py --gpu=3 --version='ko' --task=resp --log_name="HJ_C2DPR_UniInput_RAG_1e-5" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_bert --rag_our_model=c2dpr  --rag_onlyDecoderTune --hj
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="HJ_DPR_UniInput_RAG_1e-5" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_bert --rag_our_model=dpr  --rag_onlyDecoderTune --hj

# python komain.py --gpu=0 --version='ko' --task=resp --log_name="TH_Sch_128RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=20 

# python komain.py --gpu=2 --version='ko' --task=resp --log_name="HJ_C2DPR_ctx256_RAG_1e-4" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_bert --rag_our_model=c2dpr  --rag_onlyDecoderTune --hj
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="HJ_DPR_ctx256_RAG_1e-4" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_bert --rag_our_model=dpr  --rag_onlyDecoderTune --hj


# python komain.py --gpu=0 --version='ko' --task=resp --log_name="KO_Sch_128RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=20 
# python komain.py --gpu=0 --version='ko' --task=resp --log_name="KO_Sch_128RAG_know_resp측정_1e-5" --rag_lr=1e-5 --rag_epochs=20 
# python komain.py --gpu=0 --version='ko' --task=resp --log_name="KO_Sch_128RAG_know_resp측정_1e-6" --rag_lr=1e-6 --rag_epochs=20

# python komain.py --gpu=1 --version='ko' --task=resp --log_name="C2DPR_128RAG_know_resp측정_1e-4" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_model=c2dpr  --rag_onlyDecoderTune 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="C2DPR_128RAG_know_resp측정_1e-5" --rag_epochs=20 --rag_lr=1e-5 --topic_conf=0.6 --topk_topic=3 --rag_our_model=c2dpr  --rag_onlyDecoderTune 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="C2DPR_128RAG_know_resp측정_1e-6" --rag_epochs=20 --rag_lr=1e-6 --topic_conf=0.6 --topk_topic=3 --rag_our_model=c2dpr  --rag_onlyDecoderTune 

# python komain.py --gpu=3 --version='ko' --task=resp --log_name="DPR_128RAG_know_resp측정_1e-4" --rag_epochs=20 --rag_lr=1e-4 --topic_conf=0.6 --topk_topic=3 --rag_our_model=dpr  --rag_onlyDecoderTune 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="DPR_128RAG_know_resp측정_1e-5" --rag_epochs=20 --rag_lr=1e-5 --topic_conf=0.6 --topk_topic=3 --rag_our_model=dpr  --rag_onlyDecoderTune 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="DPR_128RAG_know_resp측정_1e-6" --rag_epochs=20 --rag_lr=1e-6 --topic_conf=0.6 --topk_topic=3 --rag_our_model=dpr  --rag_onlyDecoderTune 


# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_256_UniMIND_37train_37test_1e4_NoSpecialTokens" --method=unimind --uni_lr=1e-4 --uni_max_input_length=256 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="C2DPR_20Epoch_<S>256RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=20
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_Sch_<S>256RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=20 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_Sch_<S>256RAG_know_resp측정_1e-5" --rag_lr=1e-5 --rag_epochs=20 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_Sch_<S>256RAG_know_resp측정_1e-6" --rag_lr=1e-6 --rag_epochs=20 

# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_5e-5" --rag_lr=5e-5 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-5_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_70_RAG_1e-5_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-5 --rag_epochs=10 



# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_128_BART_37train_37test_1e4" --method=bart --uni_lr=1e-4 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_128_BART_37train_37test_1e6" --method=bart --uni_lr=1e-6 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_128_BART_37train_37test_1e5" --method=bart --uni_lr=1e-5 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_256_BART_37train_37test_1e4" --method=bart --uni_lr=1e-4 --uni_max_input_length=256 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_256_BART_37train_37test_1e6" --method=bart --uni_lr=1e-6 --uni_max_input_length=256 
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_256_BART_37train_37test_1e5" --method=bart --uni_lr=1e-5 --uni_max_input_length=256 

# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_128_UniMIND_37train_37test_1e5" --method=unimind --uni_lr=1e-5 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_128_UniMIND_37train_37test_1e4" --method=unimind --uni_lr=1e-4 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_128_UniMIND_37train_37test_1e6" --method=unimind --uni_lr=1e-6 --uni_max_input_length=128 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_256_UniMIND_37train_37test_1e5" --method=unimind --uni_lr=1e-5 --uni_max_input_length=256 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_256_UniMIND_37train_37test_1e4" --method=unimind --uni_lr=1e-4 --uni_max_input_length=256 
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_256_UniMIND_37train_37test_1e6" --method=unimind --uni_lr=1e-6 --uni_max_input_length=256 



# ------------------------------------------- 230729_22:00 실행시켜놓은것 아래 9개
# 아래 3개: (KO) RAG scratch 에서 knowledge retrieve점수와, resp점수까지 같이 뽑아봄
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-5" --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-6" --rag_lr=1e-6 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-4_OnlyDecoderTune" --rag_lr=1e-4 --rag_epochs=10 --rag_onlyDecoderTune
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-5_OnlyDecoderTune" --rag_lr=1e-5 --rag_epochs=10 --rag_onlyDecoderTune
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-6_OnlyDecoderTune" --rag_lr=1e-6 --rag_epochs=10 --rag_onlyDecoderTune

# 아래 3개: (KO) RAG OUR DPR 에서 resp점수 뽑아봄
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-4_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-5_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-6_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-6 --rag_epochs=10 

# # 아래 3개: (KO) RAG OUR C2DPR 에서 resp점수 뽑아봄
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_70_RAG_1e-4_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_70_RAG_1e-5_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_70_RAG_1e-6_DecTune"  --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-6 --rag_epochs=10 





# ------------------------------------------- 230729_22:00 실행시켜놓은것 아래 9개
# 아래 3개: (KO) RAG scratch 에서 knowledge retrieve점수와, resp점수까지 같이 뽑아봄
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-4" --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-5" --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=1 --version='ko' --task=resp --log_name="KO_Sch_RAG_know_resp측정_1e-6" --rag_lr=1e-6 --rag_epochs=10 

# 아래 3개: (KO) RAG OUR DPR 에서 resp점수 뽑아봄
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-4" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-5" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=2 --version='ko' --task=resp --log_name="KO_DPR_RAG_1e-6" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=dpr  --rag_lr=1e-6 --rag_epochs=10 

# 아래 3개: (KO) RAG OUR C2DPR 에서 resp점수 뽑아봄
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_RAG_1e-4" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-4 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_RAG_1e-5" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-5 --rag_epochs=10 
# python komain.py --gpu=3 --version='ko' --task=resp --log_name="KO_C2DPR_RAG_1e-6" --rag_onlyDecoderTune --rag_our_bert --rag_our_model=c2dpr  --rag_lr=1e-6 --rag_epochs=10 



# ------------------------------------------- Before 230729_16:00
# python kers_main.py --gpu=3 --version=ko --method=kers --do_pretrain --task=resp --bert_name='skt/kobert-base-v1' --log_name="KOKERS_37train_37test_1e-5" --num_epochs=15 --lr=1e-5

# python unimind_main.py --version=2 --gpu=2 --method=bart --uni_lr=5e-6 --uni_model_name='facebook/bart-large' --uni_max_input_length=128 --uni_max_target_length=128 # BART-Large 다시 돌려보기
# python unimind_main.py --version=2 --gpu=1 --method=bart --uni_lr=1e-5 --uni_model_name='facebook/bart-large' --uni_max_input_length=128 --uni_max_target_length=128 --log_name="BART_Large_37train_37test_1e-5" # BART-Large 다시 돌려보기
# python komain.py --gpu=0 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_37train_37test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune
# python main.py --gpu=1 --rag_our_model=C2DPR --task=resp --log_name="C2DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune

# python lm_main_THpaper.py --gpu=3 --log_name="BART-large_3711Train_3711Test" --num_epochs=15 --model_name='facebook/bart-large' 
# python lm_main_THpaper.py --gpu=2 --log_name="BART-large_3711Train_3711Test_1e-6" --num_epochs=15 --model_name='facebook/bart-large' --lr=1e-6


# python kers_main.py --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Retrieve_1e-5"  --lr=1e-5 --gpu=1  ## kers retrieve task
# python kers_main.py --version='ko' --bert_name='skt/kobert-base-v1' --log_name="KERS_Retrieve_1e-4"  --lr=1e-4 --gpu=2  ## kers retrieve task

# python gpt_main.py --version='2' --log_name='GPT_37_37_1e-5' --gpt_lr=1e-5 --gpu=3
# python gpt_main.py --version='ko' --bert_name='skt/kobert-base-v1' --gpt_model_name='skt/kogpt2-base-v2' --log_name='GPT_37_37_1e-6' --gpt_lr=1e-6 --gpu=3
# python gpt_main.py --version='ko' --bert_name='skt/kobert-base-v1' --gpt_model_name='skt/kogpt2-base-v2' --log_name='GPT_37_37_1e-5' --gpt_lr=1e-5 --gpu=2
# python gpt_main.py --version='ko' --bert_name='skt/kobert-base-v1' --gpt_model_name='skt/kogpt2-base-v2' --log_name='GPT_37_37_1e-4' --gpt_lr=1e-4 --gpu=1
# python gpt_main.py --version='ko' --bert_name='skt/kobert-base-v1' --gpt_model_name='kakaobrain/kogpt' --log_name='KAKAOGPT3_37_37_1e-5' --gpt_batch_size=2 --gpt_lr=1e-5 --gpu=1 

# python main.py --gpu=1 --log_name='GT_train_save' --task='goal_topic' 
# python main.py --gpu=2 --log_name='GT3711_train_save' --task='goal_topic' 

# #============================================#
# # Korean 230727 UniMIND 실험
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_BART_37train_37test_1e4" --method=bart --uni_lr=1e-4 # 512 들어가던 시절
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_BART_37train_37test_1e6" --method=bart --uni_lr=1e-6
# python bart_unimind_main_ko.py --gpu=1 --log_name="2type_BART_37train_37test_1e5" --method=bart --uni_lr=1e-5

# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_UniMIND_37train_37test_1e5" --method=unimind --uni_lr=1e-5
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_UniMIND_37train_37test_1e4" --method=unimind --uni_lr=1e-4
# python bart_unimind_main_ko.py --gpu=2 --log_name="2type_UniMIND_37train_37test_1e6" --method=unimind --uni_lr=1e-6


# python komain.py --gpu=2 --version=ko --task='goal_topic' --log_name="ko_GoalTopic_1e-5"  --lr=1e-5 --num_epochs=15
# python komain.py --gpu=2 --version=ko --task='goal_topic' --log_name="ko_GoalTopic_1e-4"  --lr=1e-4 --num_epochs=15
# python komain.py --gpu=3 --version=ko --task='goal_topic' --log_name="ko_GoalTopic_1e-6"  --lr=1e-6 --num_epochs=15


# #============================================#
# # 230722 UniMIND 실험
# python main.py --gpu=2 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune
# python main.py --gpu=3 --rag_our_model=C2DPR --task=resp --log_name="C2DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=15 --rag_our_bert --rag_onlyDecoderTune

# #============================================#
# # 230722 UniMIND 실험
# python unimind_main.py --gpu=1 --log_name="Uni_Alltrain_Alltest" --uni_train_alltype --uni_test_alltype 
# python unimind_main.py --gpu=2 --log_name="Uni_Alltrain_3711test" --uni_train_alltype 
# python unimind_main.py --gpu=3 --log_name="Uni_3711train_3711test" --uni_train_alltype 


# #============================================#
# # 230720 DPR-RAG 실험
# python lm_main_THpaper.py --gpu=3 --log_name="BART-base_AllTrain_AllTest"    --num_epochs=5 --model_name='facebook/bart-base'  --train_alltype --test_alltype 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-base_AllTrain_3711Test"   --num_epochs=5 --model_name='facebook/bart-base'  --train_alltype 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-base_3711Train_3711Test"  --num_epochs=5 --model_name='facebook/bart-base' 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-large_AllTrain_AllTest"   --num_epochs=5 --model_name='facebook/bart-large'  --train_alltype --test_alltype 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-large_AllTrain_3711Test"  --num_epochs=5 --model_name='facebook/bart-large'  --train_alltype 
# python lm_main_THpaper.py --gpu=3 --log_name="BART-large_3711Train_3711Test" --num_epochs=5 --model_name='facebook/bart-large' 


# python main.py --gpu=2 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype 
# python main.py --gpu=3 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_alltrain_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --rag_our_model=DPR --task=resp --log_name="DPR_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 

# # python main.py --gpu=0 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# # python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 

# python main.py --gpu=3 --task=resp --log_name="OUR_RAGTUNE_3711train_3711test_1e-5"   --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 

#============================================#
# 230720_18:10
# python main.py --gpu=0 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=1 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 

# python main.py --gpu=0 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 

# python main.py --gpu=3 --task=resp --log_name="OUR_RAGTUNE_3711train_3711test_1e-5"   --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 

#============================================#
# --rag_train_alltype --rag_test_alltype --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# 230719_18:25
# python main.py --gpu=0 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=0 --task=resp --log_name="Sch_RAG_alltrain_alltest_1e-6"  --rag_lr=1e-6 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=0 --task=resp --log_name="Sch_RAG_alltrain_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 

# python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-5"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_alltest_1e-6"   --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-6 --rag_epochs=5 --rag_train_alltype --rag_test_alltype
# python main.py --gpu=1 --task=resp --log_name="OUR_RAG_alltrain_3711test_1e-5"  --rag_onlyDecoderTune --rag_our_bert  --rag_lr=1e-5 --rag_epochs=5 --rag_train_alltype 

# python main.py --gpu=2 --task=resp --log_name="Sch_RAG_alltrain_3711test_1e-6"  --rag_lr=1e-6 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="Sch_RAG_3711train_3711test_1e-5"  --rag_lr=1e-5 --rag_epochs=5 
# python main.py --gpu=2 --task=resp --log_name="Sch_RAG_3711train_3711test_1e-6"  --rag_lr=1e-6 --rag_epochs=5 

# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_alltrain_3711test_1e-6"  --rag_onlyDecoderTune --rag_our_bert --rag_lr=1e-6 --rag_epochs=5 --rag_train_alltype 
# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_3711train_3711test_1e-5" --rag_onlyDecoderTune --rag_our_bert  --rag_lr=1e-5 --rag_epochs=5 
# python main.py --gpu=2 --task=resp --log_name="OUR_RAG_3711train_3711test_1e-6" --rag_onlyDecoderTune --rag_our_bert  --rag_lr=1e-6 --rag_epochs=5 

#============================================#
# 230715
# python main.py --task=resp --log_name="OUR_RAG_1e-4" --gpu=3 --rag_lr=1e-4 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_5e-4" --gpu=3 --rag_lr=5e-4 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_1e-5" --gpu=3 --rag_lr=1e-5 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_5e-5" --gpu=3 --rag_lr=5e-5 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_1e-6" --gpu=3 --rag_lr=1e-6 --rag_onlyDecoderTune
# python main.py --task=resp --log_name="OUR_RAG_5e-6" --gpu=3 --rag_lr=5e-6 --rag_onlyDecoderTune
# echo ""
# echo "RAG에서 우리 Retriever는 freeze하고 Deocder만 학습시켜본 test"
# echo "END"
#============================================#
# 230715
# python main.py --task=resp --log_name="Sch_RAG_1e-4" --gpu=0 --rag_lr=1e-4 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_5e-4" --gpu=0 --rag_lr=5e-4 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_1e-5" --gpu=0 --rag_lr=1e-5 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_5e-5" --gpu=0 --rag_lr=5e-5 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_1e-6" --gpu=0 --rag_lr=1e-6 --rag_scratch --rag_max_input_length=256
# python main.py --task=resp --log_name="Sch_RAG_5e-6" --gpu=0 --rag_lr=5e-6 --rag_scratch --rag_max_input_length=256
# echo ""
# echo "RAG에서 우리 Retriever는 freeze하고 Deocder만 학습시켜본 test"
# echo "END"
#============================================#
## 230718
# python main.py --task=resp --log_name="OUR_RAG_OnlyDec_1e-4" --gpu=3 --rag_lr=1e-4 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_OnlyDec_5e-4" --gpu=3 --rag_lr=5e-4 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_OnlyDec_1e-5" --gpu=3 --rag_lr=1e-5 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_OnlyDec_5e-5" --gpu=3 --rag_lr=5e-5 --rag_onlyDecoderTune --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_1e-4" --gpu=3 --rag_lr=1e-4  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_5e-4" --gpu=3 --rag_lr=5e-4  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_1e-5" --gpu=3 --rag_lr=1e-5  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_5e-5" --gpu=3 --rag_lr=5e-5  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_1e-6" --gpu=3 --rag_lr=1e-6  --rag_our_bert --rag_epochs=5
# python main.py --task=resp --log_name="OUR_RAG_AllTune_5e-6" --gpu=3 --rag_lr=5e-6  --rag_our_bert --rag_epochs=5
# echo "이제 Decoder만 튜닝하는거 확실해졌당"
#============================================#

# python kers_main.py --version='2' --TopicTask_Test_Prompt_usePredGoal --device=2 --inputWithKnowledge --gtpred --log_name="P_Goal_WithK_Train_PK_Test_GK_ShuffleK" --usePseudoTrain
# python kers_main.py --version='2' --TopicTask_Test_Prompt_usePredGoal --device=2 --inputWithKnowledge --inputWithTopic --gtpred --log_name="P_Goal_P_Topic_WithK_Train_PK_Test_GK_ShuffleK" --usePseudoTrain


#ORDER="1 2 3"
#for i in $ORDER
#for ((i=0; i<=3; i++))
#do
#    echo "Running loop $i"
#    # some instructions
#done

