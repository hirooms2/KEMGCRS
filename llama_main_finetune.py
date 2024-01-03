import os
import json
import sys
import torch
# import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, GenerationConfig, LlamaForCausalLM, LlamaTokenizer, Trainer
import transformers
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Union
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict # prepare_model_for_kbit_training,prepare_model_for_int8_training,set_peft_model_state_dict

from loguru import logger as mylogger
from datetime import datetime
from pytz import timezone

import utils
from data_utils import readDic
from collections import defaultdict
from copy import deepcopy
from evaluator_conv import ConvEvaluator, ConvEvaluator_ByType
import os
import sys
from typing import List
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from datasets import Dataset as datasetsDataset
from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl
# import wandb
# from peft import PeftModel
import numpy as np

# from utils.parser import parse_args

from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback


class Prompter(object):
    __slots__ = ("template", "_verbose", "args")

    def __init__(self, args, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.args = args
        # if not template_name:
        #     # Enforce the default here, so the constructor can be called with '' and will not break.
        #     if args.stage == "crs":
        #         template_name = "withoutCoT"
        #     elif args.stage == "quiz":
        #         template_name = "alpaca_legacy"
        file_name = os.path.join(args.home, "lora-alpaca","0_templates", f"{template_name}.json")
        # if not osp.exists(file_name):
        #     raise ValueError(f"Can't read {file_name}")
        if os.path.exists(file_name):
            with open(file_name) as fp:
                self.template = json.load(fp)
        else:
            self.template = {
                "description": "CRS recommendation template."
                ,"prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}" # 안쓰임
                ,"prompt_no_input": "Pretend you are a recommender system. I will give you a conversation between a user and you (a recommender system). \n Based on the conversation, create a system response. \n Here is the conversation: \n {instruction}" # Candidate items: \n {negItems}",
                # ,"prompt_no_input": "I will give you a conversation between a user and system (a recommender system). \n Based on the conversation, create system response. \n Here is the conversation: \n {instruction}" # Candidate items: \n {negItems}",
                ,"response_split": "### Response:"
                }
        if self._verbose: mylogger.info(f"Using prompt template {template_name}: {self.template['description']}")
        for k,v in self.template.items():
            mylogger.info(f"Tamplate: {k}: {v}")


    def generate_prompt(
            self, instruction: str, input: Union[None, str] = None, label: Union[None, str] = None,
            # isNew: bool = False,
            ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input: res = self.template["prompt_input"].format(instruction=instruction, input=input)
        else: res = self.template["prompt_no_input"].format(instruction=instruction)
        if label: 
            res = f"{res}{label}"
        # if label and self.args.isNew is True:
        #     if isNew is False:
        #         res = f"{res}\nChat about the item mentioned in a given dialog.\n{label}" # \nChat about the item mentioned in a given dialog.\n
        #     elif isNew is True:
        #         res = f"{res}\nRecommend the item (Do not recommend the items already mentioned in a given dialog).\n{label}" # \nRecommend the item (Do not recommend the items already mentioned in a given dialog).\n
        # elif label and self.args.isNew is False:
        #     if isNew is False:
        #         res = f"{res}{label}"
        #     elif isNew is True:
        #         res = f"{res}{label}"
        if self._verbose:
            mylogger.info(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip() # self.template["response_split"]: "### Response:"

class QueryEvalCallback(TrainerCallback):
    def __init__(self, args, evaluator):
        self.log_name = args.log_name
        self.mode = args.mode
        self.evaluator = evaluator

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        epoch = state.epoch
        path = os.path.join(args.output_dir, self.log_name + '_Epoch' + str(int(epoch)))
        os.makedirs(path)
        mylogger.info(f"epoch_{epoch} Finished. saved model in {path} ")
        model.save_pretrained(path)
        # trainer = kwargs['trainer']
        # logs = kwargs['logs']
        # model = kwargs['model']
        # mylogger.info("==============================Evaluate step==============================")
        # # predictions, labels = trainer.predict(trainer.eval_dataset)
        # # mylogger.info(predictions.size())
        # if 'test' in self.mode:
        #     self.evaluator.test(model, epoch)
        # model.train()
        # # mylogger.info(kwargs)
        # mylogger.info("==============================End of evaluate step==============================")

class Textdataset(Dataset):
    def __init__(self, args, instructions, labels, tokenizer, test_dataset_pred_aug):
        self.args = args
        self.instructions = instructions
        self.labels = labels
        self.tokenizer = tokenizer
        self.test_dataset_pred_aug=test_dataset_pred_aug

    def __getitem__(self, idx):
        return self.instructions[idx], self.labels[idx], self.test_dataset_pred_aug[idx]

    def __len__(self):
        return len(self.instructions)

class LLaMaEvaluator:
    def __init__(self, args, tokenizer, instructions: list = None, labels: list = None, negItems: list = None,
                 prompt_template_name: str = "", train_auged=None, test_auged=None):
        self.args = args
        self.instructions = instructions
        self.labels = labels
        self.negItems = negItems
        self.tokenizer = tokenizer  # , LlamaTokenizer.from_pretrained(self.args.base_model)
        self.prompter = Prompter(args, prompt_template_name)
        # self.new_idx = json.load(open(os.path.join(self.args.dataset_path, 'test_new_idx.json'), 'r', encoding='utf-8'))

        self.train_dataset_pred_aug=train_auged
        self.test_dataset_pred_aug=test_auged
        self.dataloader = self.prepare_dataloader()
        # self.model = self.prepare_model()

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def prepare_model(self,
                      base_model: str = "",
                      load_8bit: bool = False,
                      lora_weights: str = "",
                      server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
                      share_gradio: bool = False, ):
        base_model = self.args.base_model
        if self.args.lora_weights != "": lora_weights = self.args.lora_weights
        mylogger.info(f'prepare new model for evaluating || Model: {base_model}, lora_weights: {lora_weights}')
        model_cache_dir = os.path.join(self.args.home, 'model_cache', base_model)
        
        assert (base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
        assert os.path.exists(lora_weights), "Lora weitghts folder Not Exists!!!"

        if torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"
        
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(base_model, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map='auto', cache_dir=model_cache_dir)  # .to(self.args.device_id)
            # todo: For evaluating the PEFT model
            model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16,)
        else:
            Exception("CUDACUDA")
            model = LlamaForCausalLM.from_pretrained(base_model, device_map={"": device}, low_cpu_mem_usage=True , cache_dir=model_cache_dir)
            model = PeftModel.from_pretrained(model, lora_weights, device_map={"": device},)
        # unwind broken decapoda-research config
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit: model.half()  # seems to fix bugs for some users.

        return model

    def prepare_dataloader(self):
        self.tokenizer.padding_side = 'left'

        instructions = [self.prompter.generate_prompt(instruction=instruction) for instruction in self.instructions]
        instruction_dataset = Textdataset(self.args, instructions, self.labels, self.tokenizer, self.test_dataset_pred_aug)
        # instruction_dataset = LLM_RQ_Dataset(self.args, self.test_dataset_pred_aug, self.tokenizer, mode='test', method=self.args.method, template=self.prompter.template)
        dataloader = DataLoader(instruction_dataset, batch_size=self.args.eval_batch_size, shuffle=False)

        return dataloader

    def evaluate(self, input_ids, attention_mask, model, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams = 1, max_new_tokens=50, **kwargs):
        num_beams=self.args.num_beams
        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, num_return_sequences=num_beams, **kwargs,)

        with torch.no_grad():
            generation_output = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=max_new_tokens,)
        s = generation_output.sequences
        # output = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        # return [self.prompter.get_response(i) for i in self.tokenizer.batch_decode(s, skip_special_tokens=True)]
        return self.tokenizer.batch_decode(s[:,input_ids.size()[-1]:], skip_special_tokens=True)

    def test(self, model=None, epoch=None):
        mode='test'
        if model is None:
            model = self.prepare_model()
        # if epoch is not None:
        #     log_file = open(os.path.join(self.args.log_dir, f'{self.args.log_name}_Epoch{int(epoch)}.json'), 'a', buffering=1, encoding='UTF-8')
        #     self.args.log_file = log_file

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32": model = torch.compile(model)

        # hit, mentioned_hit, not_mentioned_hit, cnt, mentioned_cnt, not_mentioned_cnt, gen_mentioned_cnt, gen_not_mentioned_cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # hits, cnts = [0, 0, 0], [0, 0, 0]
        idx = 0
        rag_doc_scores, rag_contexts, contexts, label_gold_knowledges, label_pseudo_knowledges, top5_docs, real_resps, gen_resps, new_knows = [], [], [], [], [], [], [], [], []
        types, topics, p_topics = [], [], []
        topic_in_resps = []
        total_output=[]
        evaluatortype = ConvEvaluator_ByType(tokenizer= self.tokenizer, log_file_path=os.path.join(self.args.lora_weights, f"{self.args.time}_{epoch}_{mode}_GEN_REPORT_TYPE.txt") if mode == 'test' else None)
        evaluatorknowledge = ConvEvaluator_ByType(tokenizer= self.tokenizer)
        self.dataloader.tokenizer.padding_side = 'left'
        self.dataloader.tokenizer.truncation_side = 'left'
        for batch in tqdm(self.dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            generated_results = []
            batched_inputs = self.tokenizer(batch[0], padding=True,max_length=self.args.llama_input_maxlen, truncation=True, return_tensors="pt")
            batched_labels = self.tokenizer(batch[1], padding=True,max_length=self.args.llama_input_maxlen, truncation=True, return_tensors="pt") # truncation=True, max_length=cutoff_len, padding=False
            input_ids = batched_inputs["input_ids"].to(self.args.device_id)
            attention_mask = batched_inputs["attention_mask"].to(self.args.device_id)
            
            responses_gen = self.evaluate(input_ids, attention_mask, model, max_new_tokens=self.args.max_new_tokens, num_beams=self.args.num_beams)
            # responses = np.reshape(responses_gen, (-1, self.args.num_beams)).tolist()
            labels = batch[1]
            pred_aug = batch[2]
            

            contexts.extend(batch[0])
            real_resps.extend(batch[1])
            gen_resps.extend(responses_gen)
            batch_types = pred_aug['goal']
            types.extend(batch_types)
            topic_in_resps.extend([ ttt.lower() in rrr.lower() for ttt,rrr in zip(pred_aug['topic'], pred_aug['response'])])
            topics.extend(pred_aug['topic'])
            p_topics.extend(pred_aug['predicted_topic'][0])
            # pred_aug['target_knowledge']
            
            evaluatortype.evaluate(preds=responses_gen, labels=batch[1], types = batch_types, log=True, is_text=True)
            evaluatorknowledge.evaluate(preds=responses_gen, labels=pred_aug['target_knowledge'], types = batch_types, log=False, is_text=True)
            # if self.args.write:
            #     for i in generated_results:
            #         self.args.log_file.write(json.dumps(i, ensure_ascii=False) + '\n')
        
        total_output, output_str=[],[]
        report = evaluatortype.report()
        report_know = evaluatorknowledge.report()
        report_text = [f"NEW_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                       f"NEW_{epoch}_{mode}:  {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
        report_text_know = [f"Knowledge_{epoch}_{mode}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                            f"Knowledge_{epoch}_{mode}:  {report_know['bleu@1']:.3f},  {report_know['bleu@2']:.3f},  {report_know['bleu@3']:.3f},  {report_know['bleu@4']:.3f},  {report_know['dist@1']:.3f},  {report_know['dist@2']:.3f},  {report_know['dist@3']:.3f},  {report_know['dist@4']:.3f}"]
        output_str.extend(report_text)
        output_str.extend(report_text_know)
        total_output.append(f"BLEU: 1,2,3,4 | Dist: 1,2,3,4: {report_text[-1]}")

        report_type = evaluatortype.report_ByType()
        output_str.append(f"NEW_{epoch}_{mode:^5}_{'each_type':^21}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4, count")
        for each_type, report_type in report_type.items():
            reports_text = f"NEW_{epoch}_{mode:^5}_{each_type:^21}:  {report_type['bleu@1']:.3f},  {report_type['bleu@2']:.3f},  {report_type['bleu@3']:.3f},  {report_type['bleu@4']:.3f},  {report_type['dist@1']:.3f},  {report_type['dist@2']:.3f},  {report_type['dist@3']:.3f},  {report_type['dist@4']:.3f}, Count: {report_type['sent_cnt']}"
            output_str.append(reports_text)

        # evaluator.reset_metric()
        evaluatortype.reset_metric()
        evaluatorknowledge.reset_metric()
        _, _, resp_topic_str = evaluatortype.gen_resp_topic(self.args, real_resps=real_resps, types=types, topics=topics, gen_resps=gen_resps, topic_in_resps=topic_in_resps, p_topics=p_topics)
        #     if cnt % 100 == 0 and cnt != 0:
        #         # wandb.log({"hit_ratio": (hit / cnt)})
        #         mylogger.info("%.4f" % (hit / cnt))

        # self.args.score_file.write('%.4f\n' % (hit_ratio))
        output_str.extend(resp_topic_str)
        for i in output_str:
            mylogger.info(f"{i}")
        save_preds(self.args, contexts, real_resps, gen_resps, types, topics, p_topics)

def save_preds(args, contexts, real_resps, gen_resps, types, topics, p_topics):
    #
    log_file_name = f"IN_OUT_{args.log_name}"
    path = os.path.join(args.lora_weights, log_file_name)
    # if not os.path.exists(path): os.makedirs(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"Input and Output results {args.time}\n")
        f.write(f"Log File Name: {args.log_name} \n")
        for i, (ctx, resp, gen, type, topic, p_topic) in enumerate(zip(contexts, real_resps, gen_resps, types, topics, p_topics)):
            # if i == 700: break
            f.write(f"<<Source>>    : {ctx}\n")
            f.write(f"<<Real_resp>> : {resp}\n")
            f.write(f"<<Gen_resp>>  : {gen}\n")
            f.write(f"<<Type>>: {type}, <<Topic>>: {topic}, <<Predicted_Topic>>: {p_topic}")
            f.write(f"\n")
            f.write(f"\n========================================================\n")
    mylogger.info(f"Save in_out, generated results in {path}")
    return

def llama_finetune(args, tokenizer, evaluator,
        instructions: list = None,
        labels: list = None,
        # model/data params
        base_model: str = "",  # the only required argument
        output_dir: str = "/lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        num_epochs: int = 3, learning_rate: float = 3e-4, 
        cutoff_len: int = 256,
        val_set_size: int = 0,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj","v_proj",],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        # wandb_project: str = "",
        # wandb_run_name: str = "",
        # wandb_watch: str = "",  # options: false | gradients | all
        # wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca_legacy",  # The prompt template to use, will default to alpaca.
):
    output_dir = os.path.join(args.home, "lora-alpaca")
    base_model = args.base_model
    batch_size = args.batch_size
    train_on_inputs = args.train_on_inputs
    learning_rate = args.learning_rate
    gradient_accumulation_steps = args.num_device  # update the model's weights once every gradient_accumulation_steps batches instead of updating the weights after every batch.
    per_device_train_batch_size = batch_size // args.num_device
    resume_from_checkpoint = args.lora_weights
    cutoff_len = args.llama_input_maxlen

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        mylogger.info(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            # f"wandb_project: {wandb_project}\n"
            # f"wandb_run_name: {wandb_run_name}\n"
            # f"wandb_watch: {wandb_watch}\n"
            # f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(args, prompt_template_name)

    device_map = "auto"
    # device_map = args.device

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    mylogger.info("world_size: %d" % world_size)
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    # use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0: os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0: os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0: os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    tokenizer.truncation_side='left'
    tokenizer.padding_side='right' # Train 시 GPT계열의 padding side는 right --> Test시 left padding
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None,)
        if (result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token) :
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point, writeFlag=None):
        full_prompt = prompter.generate_prompt(
            instruction=data_point["instruction"],
            input=data_point["input"],
            label=data_point["output"]
            # data_point['isNew']
        )
        if writeFlag:
            mylogger.info("==========First Training sample==========\n")
            mylogger.info(f"{full_prompt}\n")

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token: user_prompt_len -= 1
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # could be sped up, probably
        return tokenized_full_prompt

    data = []
    for inst, lab in zip(instructions, labels):
        data.append({"instruction": inst, "input": "", "output": lab})
    # pkl
    
    first_sample = Dataset.from_pandas(pd.DataFrame([data[0]]))
    data = datasetsDataset.from_pandas(pd.DataFrame(data))

    if val_set_size > 0:
        Exception("Val Set Size ??? ")
        train_val = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
        val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))
    else:
        generate_and_tokenize_prompt(first_sample[0], True)
        train_data = data.shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    cache_dir = os.path.join(args.home, 'model_cache', base_model)
    mylogger.info(f"Train instruction length avg: {sum([len(i) for i in train_data['instruction']])/len(train_data):.1f}")
    mylogger.info(f"Train input length avg: {sum([len(i) for i in train_data['input']])/len(train_data):.1f}")
    mylogger.info(f"Train label length avg: {sum([len(i) for i in train_data['output']])/len(train_data):.1f}")
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        # load_in_8bit_fp32_cpu_offload=True,
        # llm_int8_enable_fp32_cpu_offload=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        cache_dir=cache_dir
        # quantization_config=quantization_config,
    )

    tokenizer.pad_token_id = (0)  # unk. we want this to be different from the eos token
    # tokenizer.truncation_side = 'left'
    # tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,   lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",    task_type="CAUSAL_LM", )

    model = get_peft_model(model, config)

    if resume_from_checkpoint[resume_from_checkpoint.rfind('/') + 1:] != "lora-alpaca":
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (False)  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            mylogger.info(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            mylogger.info(f"Checkpoint {checkpoint_name} not found")
    else:
        resume_from_checkpoint = None
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=5 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=args.wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[QueryEvalCallback(args, evaluator)]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))
    if torch.__version__ >= "2" and sys.platform != "win32": model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # model.save_pretrained(output_dir)
    mylogger.info("\n If there's a warning about missing keys above, please disregard :)")


def add_ours_specific_args(parser=None):
    if parser: pass
    else: parser = argparse.ArgumentParser()
    # parser.add_argument("--asdf", action='store_true', help="~할지 여부")
    parser.add_argument( "--method", type=str, default="llama", choices=["bart","unimind","t5","llm", "llama"], help=" Method " )
    parser.add_argument( "--version", type=str, default="2", choices=["2","ko"], help=" Version for directory " )
    parser.add_argument( "--data_dir", type=str, default="llama", help=" DATA Dir " )
    parser.add_argument( "--log_dir", type=str, default="logs", help=" LOG Dir " )
    parser.add_argument( "--home", type=str, default="llama", help=" HOME DIR " )
    parser.add_argument( "--debug", action='store_true', help="Whether to DEBUG MODE.")
    
    # parser.add_argument("--topic_rq", type=str, default='conf', choices=["conf","top","none"] , help=" Method ")
    
    # parser.add_argument("--llama_max_length", type=int, default=256, help=" Goal-Topic input max_length ")
    # parser.add_argument("--llama_batch_size", type=int, default=8, help=" Method ")
    # parser.add_argument("--uni_max_input_length", type=int, default=256, help=" input len: 256 ")
    # parser.add_argument("--uni_max_target_length", type=int, default=128, help=" output len: 128 ")
    # parser.add_argument("--uni_num_beams", type=int, default=1, help=" num beam ") # Only one

    parser.add_argument("--device", type=str, default='0')

    parser.add_argument('--llama_input_maxlen', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--max_dialog_len', type=int, default=128)
    parser.add_argument('--rq_num', type=str, default='1_5choice')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        choices=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf',
                                 'meta-llama/Llama-2-7b-chat-hf', 'gpt-3.5-turbo', 'google/flan-t5-large', 't5-small',
                                 't5-large'])
    parser.add_argument('--dataset_path', type=str, default='data/redial')
    parser.add_argument('--stage', type=str, default='crs')  # crs or quiz
    parser.add_argument('--model_name', type=str, default='llama')
    parser.add_argument('--num_device', type=int, default=1)
    parser.add_argument("--write", action='store_true', help="Whether to write of results.")
    parser.add_argument("--lora_path", type=str, default='lora-alpaca')
    parser.add_argument("--lora_weights", type=str, default='')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train', 'test', 'valid', 'train_test'])
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--data_type', type=str, default='default')
    # parser.add_argument('--isNew', type=bool, default=False)
    parser.add_argument('--oversample_ratio', type=int, default=1)
    parser.add_argument('--train_response', type=bool, default=False)
    parser.add_argument('--num_reviews', type=int, default=1)
    parser.add_argument('--train_on_inputs', type=bool, default=True)
    parser.add_argument('--merge', type=bool, default=False)
    parser.add_argument('--quiz_merge', type=bool, default=False)
    parser.add_argument('--origin_augment', type=bool, default=False)
    parser.add_argument('--all_merge', type=bool, default=False)
    parser.add_argument('--plot_merge', type=bool, default=False)
    return parser

def initLogging(args):
    import git ## pip install gitpython
    filename = args.log_name #f'{args.time}_{"DEBUG" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'
    filename = os.path.join(args.log_dir, filename)
    mylogger.remove()
    fmt = "<green>{time:YYMMDD_HH:mm:ss}</green> | {message}"
    # if not args.debug : 
    mylogger.add(filename, format=fmt, encoding='utf-8')
    mylogger.add(sys.stdout, format=fmt, level="INFO", colorize=True)
    mylogger.info(f"FILENAME: {filename}")
    try: mylogger.info(f"Git commit massages: {git.Repo(search_parent_directories=True).head.object.hexsha[:7]}")
    except: pass
    mylogger.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    return mylogger

def main(args=None):
    parser = argparse.ArgumentParser(description="ours_main.py")
    # parser = utils.default_parser(parser)
    parser = add_ours_specific_args(parser)
    args = parser.parse_args()
    args.home = os.path.dirname(os.path.realpath(__file__))
    args.time = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H%M%S')
    args.data_dir = os.path.join(args.home, 'data', '2')
    args.log_dir = os.path.join(args.home, 'logs', args.version, args.method)
    args.log_name = f'{args.time}_{f"DEBUG_{args.log_name}" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'  # TIME_LOGNAME_MODELNAME_log.txt
    args.device = f'cuda:{args.device}'
    args.device_id = args.device
    args.lora_path = os.path.join(args.home, args.lora_path)
    args.lora_weights = os.path.join(args.lora_path, args.lora_weights)
    initLogging(args)
    mylogger.info("Read raw file")
    topicDic , goalDic = readDic(os.path.join(args.data_dir, "topic2id.txt")), readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic, args.goalDic = topicDic, goalDic
    args.topic_num, args.goal_num = len(topicDic['int']), len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}
    train_dataset_aug_pred, test_dataset_aug_pred = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'pkl_794', f'train_pred_aug_dataset.pkl')) , utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', f'pkl_794', f'test_pred_aug_dataset.pkl'))
    if args.debug: train_dataset_aug_pred, test_dataset_aug_pred = train_dataset_aug_pred[:10], test_dataset_aug_pred[:10]
    
    train_instructions, train_labels = [i['dialog'].replace("[SEP]", "\n") for i in train_dataset_aug_pred], [i['response'].replace("[SEP]", "") for i in train_dataset_aug_pred]
    test_instructions, test_labels = [i['dialog'].replace("[SEP]", "\n") for i in test_dataset_aug_pred], [i['response'].replace("[SEP]", "") for i in test_dataset_aug_pred]

    if 'llama' in args.base_model.lower():
        cache_dir = os.path.join(args.home, 'model_cache', args.base_model)
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model, cache_dir = cache_dir)
        evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, instructions=test_instructions, labels=test_labels,
                                   prompt_template_name=args.prompt, train_auged=train_dataset_aug_pred, test_auged=test_dataset_aug_pred)
        if 'train' in args.mode:
            llama_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=train_instructions,
                           labels=train_labels, num_epochs=args.epoch, 
                           prompt_template_name=args.prompt)
        if 'test' in args.mode:
            # 특정 weight 지정 없이, 모든 epoch 에 해당하는 weights test
            # if args.lora_weights[args.lora_weights.rfind('/') + 1:] != "lora-alpaca" and args.lora_weights[-1].isdigit() is False:
            #     origin_lora_weights = args.lora_weights
            #     for e in range(args.epoch):
            #         args.lora_weights = origin_lora_weights + '_Epoch' + str(int(e + 1))
            #         evaluator.test(epoch=e + 1)
            if 'train' in args.mode: # Train끝나고 Test들어왔을 땐, 진행된 epoch 들에 대해 다 test 진행
                for e in range(args.epoch):
                    args.lora_weights = os.path.join(args.lora_path, args.log_name + '_Epoch' + str(int(e + 1)))
                    evaluator.test(epoch=e + 1)
            else: # 그렇지 않을 땐 
                if args.lora_weights[-1].isdigit() :  # default lora_weights (i.e., not-trained LLaMa)
                    mylogger.info(f"Test at {args.lora_weights}")
                # if args.lora_weights[args.lora_weights.rfind('/') + 1:] == "lora-alpaca":  # default lora_weights (i.e., not-trained LLaMa)
                    evaluator.test()
                else:
                    origin_lora_weights_epochs = args.lora_weights[args.lora_weights.rfind('/') + 1:]
                    weights_list = sorted(list(filter(lambda x: origin_lora_weights_epochs in x , os.listdir(args.lora_path))), reverse=True)[:-4]
                    # sorted(weights_list)
                    mylogger.info(f"<Weights list>: [{weights_list}]")
                    for e, lora_weight_path in enumerate(weights_list):
                        mylogger.info(f"Test at {args.lora_weights}")
                        args.lora_weights = f"{os.path.join(args.lora_path, lora_weight_path)}"
                        evaluator.test(epoch=e + 1)

                    # for e in range(args.epoch):
                    #     args.lora_weights = f"{os.path.join(args.lora_path, origin_lora_weights_epochs)}{str(int(e + 1))}"
                    #     evaluator.test(epoch=e + 1)
                        # evaluator.test()

    print("END")

if __name__ == '__main__':
    main()






