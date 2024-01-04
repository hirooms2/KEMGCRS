import json
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict
from transformers import AutoTokenizer
from utils import read_pkl
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
class ConvEvaluator:
    def __init__(self, tokenizer, log_file_path=None):
        self.tokenizer = tokenizer
        self.reset_metric()
        if log_file_path:
            self.log_file = open(log_file_path, 'w', buffering=1, encoding='utf-8')
            self.log_cnt = 0

    def evaluate(self, preds, labels, log=False):
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in decoded_preds]
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in decoded_labels]
        decoded_labels = [label.strip() for label in decoded_labels]

        if log and hasattr(self, 'log_file'):
            for pred, label in zip(decoded_preds, decoded_labels):
                self.log_file.write(json.dumps({'pred': pred, 'label': label}, ensure_ascii=False) + '\n')

        self.collect_ngram(decoded_preds)
        self.compute_bleu(decoded_preds, decoded_labels)
        self.sent_cnt += len([pred for pred in decoded_preds if len(pred) > 0])

    def collect_ngram(self, strs):
        for str in strs:
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)

    def compute_bleu(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                weights = [0] * 4
                weights[k] = 1
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            else:
                if 'dist' in k: v = len(v)
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report

    def reset_metric(self):
        self.metric = {'bleu@1': 0, 'bleu@2': 0, 'bleu@3': 0, 'bleu@4': 0,
                       'dist@1': set(), 'dist@2': set(), 'dist@3': set(), 'dist@4': set(), }
        self.sent_cnt = 0

    def after_eval_report(self, preds, labels):
        self.collect_ngram(preds)
        self.compute_bleu(preds, labels)
        self.sent_cnt += len([pred for pred in preds if len(pred) > 0])
        pass


class ConvEvaluator_ByType:
    def __init__(self, tokenizer, log_file_path=None):
        self.tokenizer = tokenizer
        self.reset_metric()
        if log_file_path:
            self.log_file = open(log_file_path, 'w', buffering=1, encoding='utf-8')
            self.log_cnt = 0

    def evaluate(self, preds, labels, types, log=False, start_id=0, is_text=False):
        if is_text: 
            decoded_preds = preds
            decoded_labels = labels
        else:
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in decoded_preds]
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in decoded_labels]
            decoded_labels = [label.strip() for label in decoded_labels]

        if log and hasattr(self, 'log_file'):
            for pred, label, type in zip(decoded_preds, decoded_labels, types):
                self.log_file.write(json.dumps({'pred': pred, 'label': label, 'type': type}, ensure_ascii=False) + '\n')

        self.collect_ngram(decoded_preds, types)
        self.compute_bleu(decoded_preds, decoded_labels, types)
        self.sent_cnt += len([pred for pred in decoded_preds if len(pred) > 0])

        for pred, type in zip(preds, types):
            self.sent_cnt_byType[type] += 1 if len(pred) > 0 else 0

    def collect_ngram(self, strs, types):
        for str, type in zip(strs, types):
            if type not in self.metric_byType: self.metric_byType[type] = {'bleu@1': 0, 'bleu@2': 0, 'bleu@3': 0, 'bleu@4': 0, 'dist@1': set(), 'dist@2': set(), 'dist@3': set(), 'dist@4': set(), }
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)
                    self.metric_byType[type][dist_k].add(token)

    def compute_bleu(self, preds, labels, types):
        # tqdm(zip(preds, labels, types), desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}')
        # for pred, label, type in tqdm(zip(preds, labels, types), desc="Compute_Bleu", bar_format='{l_bar} | {bar:23} {r_bar}'):
        for pred, label, type in zip(preds, labels, types):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                weights = [0] * 4
                weights[k] = 1
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)
                self.metric_byType[type][f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            else:
                if 'dist' in k: v = len(v)
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report

    def report_ByType(self):
        report = defaultdict(defaultdict)
        for type, type_report in self.metric_byType.items():
            if isinstance(type, str):
                for k, v in self.metric_byType[type].items():
                    if self.sent_cnt_byType[type] == 0:
                        report[type][k] = 0
                    else:
                        if 'dist' in k: v = len(v)
                        report[type][k] = v / self.sent_cnt_byType[type]
                report[type]['sent_cnt'] = self.sent_cnt_byType[type]
        return report

    def reset_metric(self):
        self.metric_byType = defaultdict(defaultdict)
        self.sent_cnt_byType = defaultdict(int)

        self.metric = {'bleu@1': 0, 'bleu@2': 0, 'bleu@3': 0, 'bleu@4': 0, 'dist@1': set(), 'dist@2': set(), 'dist@3': set(), 'dist@4': set(), }
        self.sent_cnt = 0

    def after_eval_report(self, preds, labels, types):
        self.collect_ngram(preds, types)
        self.compute_bleu(preds, labels, types)
        self.sent_cnt += len([pred for pred in preds if len(pred) > 0])
        pass
    
    @staticmethod
    def gen_resp_topic(args, real_resps=None, types=None, topics=None, gen_resps=None, topic_in_resps=None, p_topics=None, isrq=False): # For HitGen
        typelist = ['Q&A', 'Movie recommendation', 'Music recommendation', 'POI recommendation', 'Food recommendation'] if args.version != 'ko' else ['QA', 'Movie Recommendation']
        hitdic = {type: {'hit1_Rec': 0, 'hit1_Gen': 0, 'total': 0} for type in typelist + ['Others', 'total']}
        for idx in range(len(real_resps)):
            goal_type = types[idx]
            if goal_type in typelist:
                tmp_goal = goal_type
            else:
                tmp_goal = 'Others'

            pred, gold, topic, p_topic = gen_resps[idx].lower(), real_resps[idx].lower(), topics[idx].lower(), p_topics[idx].lower()
            topic_in_resp = topic_in_resps[idx]
            if topic in gold:
                try:
                    if isrq: 
                        pred, gold = pred.split('|')[-1] , gold.split('|')[-1]
                except: pass

                if topic_in_resp:
                    hitdic['total']['total'] += 1
                    hitdic[tmp_goal]['total'] += 1
                    if topic in pred:
                        hitdic[tmp_goal]['hit1_Gen'] += 1
                        hitdic['total']['hit1_Gen'] += 1
                    if topic == p_topic:
                        hitdic[tmp_goal]['hit1_Rec'] += 1
                        hitdic['total']['hit1_Rec'] += 1
                
        hitdic_ratio = {goal_type: {'hit1_Rec': 0, 'hit1_Gen': 0, 'total': 0} for goal_type in typelist + ["Others", 'total']}
        output_str = [f"  HITGEN:                     hit1_Rec,  hit1_Gen,  total_cnt"]
        for key in hitdic.keys():
            if hitdic[key]['total']:
                hitdic_ratio[key]['hit1_Gen'] = hitdic[key]['hit1_Gen'] / hitdic[key]['total']
                hitdic_ratio[key]['hit1_Rec'] = hitdic[key]['hit1_Rec'] / hitdic[key]['total']

            hitdic_ratio[key]['total'] = hitdic[key]['total']
            output_str.append(f"HITGEN: {key:^22}: {hitdic_ratio[key]['hit1_Rec']:.3f}, {hitdic_ratio[key]['hit1_Gen']:.3f}, {hitdic_ratio[key]['total']}")
        output_str.append(f"(pred) Topic Hit Ratio: {sum([p == g for p, g in zip(p_topics, topics)]) / len(p_topics):.3f}")
        return hitdic, hitdic_ratio, output_str
class Args:
    def __init__(self, version) -> None:
        self.version = version

def conv_gen_eval(version='2', model_result='bartbase', when='231229', fixedPath=None):
    import os
    # home, bert_name = '/home/work/CRSTEST/KEMGCRS/', 'bert-base-uncased'
    
    home, bert_name = os.path.dirname(os.path.realpath(__file__)) , 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(bert_name, cache_dir=os.path.join(home, "model_cache", bert_name))
    evaluator = ConvEvaluator_ByType(tokenizer=tokenizer, log_file_path=None)
    evaluator_knowledgebleu = ConvEvaluator_ByType(tokenizer=tokenizer, log_file_path=None)
    test_dataset= read_pkl('/home/work/CRSTEST/KEMGCRS/data/2/pred_aug/pkl_794/test_pred_aug_dataset.pkl') # Data for true type, topic, knowledges
    knowledges3711=[i['target_knowledge'] for i in test_dataset]
    
    rep_path = os.path.join(home, 'temp_code', 'hitgen', version, when,model_result) # "/home/work/CRSTEST/KERS_HJ/epoch_output/2/2023-07-23_052257_BKERS_3711Train_3711Test_1e-5_facebook_bart-base/12_test_GEN_REPORT.txt"
    if fixedPath: rep_path = fixedPath

    types, preds, labels = [], [], []
    with open(rep_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            linedic = json.loads(line)
            predtxt=""
            if 'User: ' in linedic['pred']: 
                predtxt = linedic['pred'].split('User')[0]
            else: predtxt = linedic['pred']
            
            if predtxt[:8]!='System: ' and linedic['label'][:8]=='System: ': 
                preds.append(f"System: {predtxt}")
            else: 
                preds.append(linedic['pred'])
            labels.append(linedic['label'])
            types.append(linedic['type'])
    evaluator.after_eval_report(preds[:], labels[:], types[:])
    evaluator_knowledgebleu.after_eval_report(preds[:], knowledges3711[:], types[:])
    report = evaluator.report()
    report_knowledgebleu = evaluator_knowledgebleu.report()
    report_text = [f" {model_result}: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                   f"{model_result}: {report['bleu@1']:.3f},  {report['bleu@2']:.3f},  {report['bleu@3']:.3f},  {report['bleu@4']:.3f},  {report['dist@1']:.3f},  {report['dist@2']:.3f},  {report['dist@3']:.3f},  {report['dist@4']:.3f}"]
    report_text_knowledgebleu = [f" {model_result}_knowledge_bleu: bleu@1, bleu@2, bleu@3, bleu@4, dist@1, dist@2, dist@3, dist@4",
                   f"{model_result}_knowledge_bleu: {report_knowledgebleu['bleu@1']:.3f},  {report_knowledgebleu['bleu@2']:.3f},  {report_knowledgebleu['bleu@3']:.3f},  {report_knowledgebleu['bleu@4']:.3f},  {report_knowledgebleu['dist@1']:.3f},  {report_knowledgebleu['dist@2']:.3f},  {report_knowledgebleu['dist@3']:.3f},  {report_knowledgebleu['dist@4']:.3f}"]
    print(report_text[0], '\n', report_text[1])
    print(report_text_knowledgebleu[0], '\n', report_text_knowledgebleu[1])

    topics = [i['topic'] for i in test_dataset]
    topic_in_resps = [i['topic'].lower() in i['response'].lower() for i in test_dataset]
    p_topics = [i['predicted_topic'][0] for i in test_dataset]
    
    hitdic, hitdic_ratio, output_str = evaluator.gen_resp_topic(Args(version), real_resps=labels, types=types, topics=topics, gen_resps=preds, topic_in_resps=topic_in_resps, p_topics=p_topics)
    
    for i in output_str: 
        print(i)


def know_hit_ratio(args, pred_pt, gold_pt, new_knows=None, types=None, typelist=['Q&A', 'Movie recommendation', 'Music recommendation', 'POI recommendation', 'Food recommendation'], new_check=False):
    if args.version == 'ko': typelist = ['QA', 'Movie Recommendation']
    hitdic = {type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit10': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'hit10_new': 0, 'total': 0} for type in typelist + ['Others', 'total']}
    for idx in range(len(gold_pt)):
        goal_type = types[idx]
        if goal_type in typelist: tmp_goal = goal_type
        else: tmp_goal = 'Others'

        pred, gold = pred_pt[idx], gold_pt[idx]

        hitdic[tmp_goal]['total'] += 1
        hitdic['total']['total'] += 1

        if gold in pred:
            hitdic[tmp_goal]['hit10'] += 1
            hitdic['total']['hit10'] += 1
            if gold in pred[:5]:
                hitdic[tmp_goal]['hit5'] += 1
                hitdic['total']['hit5'] += 1
                if gold in pred[:3]:
                    hitdic[tmp_goal]['hit3'] += 1
                    hitdic['total']['hit3'] += 1
                    if gold == pred[0]:
                        hitdic[tmp_goal]['hit1'] += 1
                        hitdic['total']['hit1'] += 1

        if new_knows and new_check:
            if new_knows[idx]:
                if gold in pred:
                    hitdic[tmp_goal]['hit10'] += 1
                    hitdic['total']['hit10'] += 1
                    if gold in pred[:5]:
                        hitdic[tmp_goal]['hit5'] += 1
                        hitdic['total']['hit5'] += 1
                        if gold in pred[:3]:
                            hitdic[tmp_goal]['hit3'] += 1
                            hitdic['total']['hit3'] += 1
                            if gold == pred[0]:
                                hitdic[tmp_goal]['hit1'] += 1
                                hitdic['total']['hit1'] += 1


    hitdic_ratio = {goal_type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit10': 0, 'total': 0} for goal_type in typelist + ["Others", 'total']}
    # hitdic_ratio = {goal_type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'hit10': 0, 'hit1_new': 0, 'hit3_new': 0, 'hit5_new': 0, 'hit10_new': 0, 'total': 0} for goal_type in typelist + ["Others", 'total']}
    # output_str = [f"                         hit1,  hit3,  hit5, hit10, hit1_new, hit3_new, hit5_new, hit10_new, total_cnt"]
    output_str = [f"                             hit1,  hit3,  hit5,  hit10,  total_cnt"]
    for key in hitdic.keys():
        hit_lists= ['hit1', 'hit3', 'hit5', 'hit10']
        for hit in hit_lists:
            if hitdic[key]['total']:
                hitdic_ratio[key][hit] = hitdic[key][hit] / hitdic[key]['total']
        hitdic_ratio[key]['total'] = hitdic[key]['total']
        output_str.append(f"{key:^25}: {hitdic_ratio[key]['hit1']:.3f}\t{hitdic_ratio[key]['hit3']:.3f}\t{hitdic_ratio[key]['hit5']:.3f}\t{hitdic_ratio[key]['hit10']:.3f}\t{hitdic_ratio[key]['total']}")
        # output_str.append(f"{key:^22}: {hitdic_ratio[key]['hit1']:.3f}\t{hitdic_ratio[key]['hit3']:.3f}\t{hitdic_ratio[key]['hit5']:.3f}\t{hitdic_ratio[key]['hit10']:.3f}\t{hitdic_ratio[key]['hit1_new']:.3f}\t{hitdic_ratio[key]['hit3_new']:.3f}\t{hitdic_ratio[key]['hit5_new']:.3f}\t{hitdic_ratio[key]['hit10_new']:.3f}\t{hitdic_ratio[key]['total']}")
    return hitdic, hitdic_ratio, output_str

if __name__ == '__main__':
    import json

    # conv_gen_eval(version='2', model_result='bartbase', when='231229')
    conv_gen_eval(version='2', model_result='kers_base_10', when='231229')
    pass