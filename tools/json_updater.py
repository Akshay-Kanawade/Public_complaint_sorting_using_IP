import json


def get_json_data():
    with open("complaint_information.json") as fp:
        data = json.load(fp)
    return data


def update_json(user_data, json_file=get_json_data()):
    json_file[user_data['class']].append(user_data)
    with open("complaint_information.json", 'w') as f:
        json.dump(json_file, f)
