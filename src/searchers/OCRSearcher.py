import os
from fuzzywuzzy import process, fuzz
import json
import time
from tqdm import tqdm

# Hàm kiểm tra từ gần giống query trong câu


def find_closest_match(query, sentences, threshold=60):
    tokens = sentences.split('\n')
    for token in tokens:
        # tìm đúng chính xác query đó k sai 1 kí tự, nếu ko ra thì mới so độ tương đồng lấy gần đúng.
        if query.lower() in token.lower().replace('_', ' '):
            return (100, token)

        similarity = fuzz.ratio(query.lower(), token.lower())
        if similarity >= threshold:
            return (similarity, token)
        for word in token.split():
            simi = fuzz.ratio(query.lower(), word.lower().replace(
                '_', ' '))  # word.lower().replace('_',' ')
            if simi >= threshold:
                return (simi, token)
    return None


def search_compare_similirity_word_load_fulldatabase_to_ram(query, database, num_img=10):
    results = []
    for vid, data in database.items():
        for k, v in data.items():
            check = find_closest_match(query, v)
            if check is not None:
                results.append(
                    {"video_name": vid[:-5], "keyframe_id": int(k[:-4]), "score": check[0]})
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    # print(results_sorted[:num_img])
    return results_sorted[:num_img]
