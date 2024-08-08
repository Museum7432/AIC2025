import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json
import time
from tqdm import tqdm

# Hàm kiểm tra từ gần giống query trong câu
def find_closest_match(query, sentence, threshold=60):
    tokens = sentence.split('\n')
    # words.append(sentence.replace('\n',' '))
    for token in tokens:
            if query.lower() in token.lower().replace('_',' '):# tìm đúng chính xác query đó k sai 1 kí tự, nếu ko ra thì mới so độ tương đồng lấy gần đúng.
                return (100, token)
                # return
            # Tính toán độ giống nhau giữa từ truy vấn và từng từ trong câu
            similarity = fuzz.ratio(query.lower(), token.lower())
            if similarity >= threshold:
                # print(similarity)
                return (similarity, token)
                # return True
            #Phân vân có nên tách từ trong database ko.
            for word in token.split():
                simi = fuzz.ratio(query.lower(), word.lower().replace('_',' '))#word.lower().replace('_',' ')
                if simi >= threshold:
                    # print(simi)
                    return (simi, token)
                    # return True
    # return False
    return None

def search_compare_similirity_word(query):# ko load data lên ram trước
    results=[]
    for vid in os.listdir(PATH_TO_DB):
        path_to_file=os.path.join(PATH_TO_DB,vid)
        with open(path_to_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k,v in data.items():
            if find_closest_match(query,v):
                results.append(os.path.join(vid[:-5],k))
    return  results
def search_compare_similirity_word_load_fulldatabase_to_ram(query,database,num_img=10):
    results=[]
    for vid,data in database.items():
        for k,v in data.items():
            check = find_closest_match(query,v)
            if check is not None:
                results.append({"video_name": vid[:-5], "keyframe_id": int(k[:-4]),"score": check[0]})
    results_sorted = sorted(results, key=lambda x:x["score"], reverse=True)
    # print(results_sorted[:num_img])
    return  results_sorted[:num_img]