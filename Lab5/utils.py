import torch
import json

def test_obj_getter(obj_json_path = "objects.json", test_json_path = 'test.json'):
    with open(obj_json_path, "r") as objfile:
        class_dict = json.load(objfile)
    with open(test_json_path, "r") as testfile:
        test_list = json.load(testfile)
    
    labels_encoding = torch.zeros(len(test_list), len(class_dict))
    for i in range(len(test_list)):
        for obj in test_list[i]:
            labels_encoding[i, int(class_dict[obj])] = 1.0

    return labels_encoding

