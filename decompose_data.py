import pickle
from tqdm import tqdm
import csv

train_data = pickle.load(open('data/raw/us-filtered-asin-train-data.pkl', 'rb'))
test_data = pickle.load(open('data/raw/us-filtered-asin-test-data.pkl', 'rb'))

train_csv = []
asin_csv = []
exist_asin = set()
for i, session in tqdm(enumerate(train_data)):
    for action in session:
        train_csv.append([i, action[0], action[1], action[2], action[3]])
        if action[3] is not None:
            if action[3] not in exist_asin:
                exist_asin.add(action[3])
                asin_csv.append([action[3], action[4], action[5], action[6]])

test_csv = []
for i, session in tqdm(enumerate(test_data)):
    for action in session:
        test_csv.append([i+len(train_data), action[0], action[1], action[2], action[3]])
        if action[3] is not None:
            if action[3] not in exist_asin:
                exist_asin.add(action[3])
                asin_csv.append([action[3], action[4], action[5], action[6]])

with open('data/us-filtered-train.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['session_id', 'timestamp', 'action type', 'keyword', 'asin'])
    write.writerows(train_csv)


with open('data/us-filtered-test.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['session_id', 'timestamp', 'action type', 'keyword', 'asin'])
    write.writerows(test_csv)


with open('data/us-filtered-asin.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['asin', 'product type', 'brand', 'product title'])
    write.writerows(asin_csv)


