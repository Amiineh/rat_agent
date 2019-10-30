import os
import json


def run(id, output_path):
    info_path = output_path + 'train.json'
    with open(info_path) as infile:
        info = json.load(infile)

    found = False
    for x in info:
        if info[str(x)]['id'] == id:
            print("removing id {}: {}".format(id, info[str(x)]))
            del info[str(x)]
            found = True
            break

    if not found:
        print("id {} not found, it may have been deleted or never existed".format(id))

    with open(info_path, 'w') as outfile:
        json.dump(info, outfile)