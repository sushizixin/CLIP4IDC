import json
import os

data_path = "your path of the spot-the-diff data"    ### your path of spot-the-diff data

train_data = json.load(open(os.path.join(data_path, 'train.json'), 'r'))
val_data = json.load(open(os.path.join(data_path, 'val.json'), 'r'))
test_data = json.load(open(os.path.join(data_path, 'test.json'), 'r'))


def reformat(data):
    new_data = {}
    for d in data:
        d_id = d["img_id"]
        d_sent = d["sentences"]
        if d_id not in new_data:
            new_data[d_id] = d_sent
        else:
            new_data[d_id].extend(d_sent)

    new_data_list = []
    for k, v in new_data.items():
        item = {"img_id": k, "sentences": v}
        new_data_list.append(item)

    return new_data_list


def save_json(data, filename):
    json.dump(data, open(filename, 'w'))


reformat_train_data = reformat(train_data)
reformat_val_data = reformat(val_data)
reformat_test_data = reformat(test_data)

save_json(reformat_train_data, os.path.join(data_path, "reformat_train.json"))
save_json(reformat_val_data, os.path.join(data_path, "reformat_val.json"))
save_json(reformat_test_data, os.path.join(data_path, "reformat_test.json"))
