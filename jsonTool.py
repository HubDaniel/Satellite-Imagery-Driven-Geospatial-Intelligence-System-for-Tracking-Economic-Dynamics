import json
import os
import csv

def outputJson(jsonObj, file):
    print('========Starting to save file:', file)
    with open(file, 'w') as f:
        json.dump(jsonObj, f, indent=4, ensure_ascii=False)
        f.flush()
        f.close()
    print('=================Save successful')

def inputJson(file):
    jsonObj = ''
    with open(file, 'r') as f:
        jsonObj = json.load(f)
        f.close()
    return jsonObj   

def read_csv(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, mode='r', encoding='utf-8') as f:
            lines = csv.reader(f)  # This reads each line of data as a list
            for line in lines:
                data.append(line)
        return data
    else:
        print('Filepath is incorrect:', filepath)
        return []   

# Read csv
def write_csv(filepath, data, head=None):
    if head:
        data = [head] + data
    with open(filepath, mode='w', encoding='UTF-8-sig', newline='') as f:
        writer = csv.writer(f)
        for i in data:
            writer.writerow(i)


if __name__ == '__main__':
    outputJson({"t": 'test'}, './data/temp/t.json')
    t = inputJson('./data/temp/t.json')
    print('t:', t)
