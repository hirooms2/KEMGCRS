import pickle
import json
import random
from tqdm import tqdm 

raw = pickle.load(open("/home/user/junpyo/KEMGCRS/data/2/pred_aug/pkl_794/test_pred_aug_dataset.pkl", 'rb'))

pred = json.load(open("/home/user/junpyo/KEMGCRS/data/2/pred_aug/know/our/cotmae/en_test_know_3711_copy.txt", 'r', encoding='utf-8'))

test_know = json.load(open("/home/user/junpyo/KEMGCRS/all_knowledgeDB.json", 'r', encoding='utf-8'))

new_pred_list = []

# for i, j in zip(raw, pred):
#     target_knowledge = i['target_knowledge']
#     candidates = [target_knowledge]
#     while len(candidates) < 5:
#         selected = random.choice(test_know)
#         if selected not in candidates:
#             candidates.append(selected)
#     new_pred_list.append({'predicted_know': candidates, 'predicted_know_conf': [1] * len(candidates)})

# f = open("/home/user/junpyo/KEMGCRS/en_test_know_3711_random.txt", "w+")
# for i in new_pred_list: # range(len(new_pred_list)):
#     f.write(json.dumps(i, ensure_ascii=False) + '\n')
# f.close()
# # json.dump(new_pred_list, open('/home/user/junpyo/KEMGCRS/en_test_know_3711_random.json', 'w'), indent=4)
# # print()

for i, j in tqdm(zip(raw, pred)):
    target_knowledge = i['target_knowledge']
    candidates = [target_knowledge]
    topic = i['topic']
    topic_related_know = [i for i in test_know if topic.lower() in i.lower()]

    while len(candidates) < min(5, len(topic_related_know)):
        selected = random.choice(topic_related_know)
        if selected not in candidates:
            candidates.append(selected)

    while len(candidates) < 5:
        selected = random.choice(test_know)
        if selected not in candidates:
            candidates.append(selected)
    new_pred_list.append({'predicted_know': candidates, 'predicted_know_conf': [1] * len(candidates)})

f = open("/home/user/junpyo/KEMGCRS/en_test_know_3711_random_sametopic.txt", "w")
for i in new_pred_list: # range(len(new_pred_list)):
    f.write(json.dumps(i, ensure_ascii=False) + '\n')
f.close()
# json.dump(new_pred_list, open('/home/user/junpyo/KEMGCRS/en_test_know_3711_random_sametopic.json', 'w'), indent=1)