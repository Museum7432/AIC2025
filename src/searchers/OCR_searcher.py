import os
from fuzzywuzzy import process, fuzz
import json
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from database import OcrDB
# Hàm kiểm tra từ gần giống query trong câu


def find_closest_match(query, sentences, threshold=55):
    tokens = sentences.split("\n")
    for token in tokens:
        # tìm đúng chính xác query đó k sai 1 kí tự, nếu ko ra thì mới so độ tương đồng lấy gần đúng.
        if query.lower() in token.lower().replace("_", " "):
            return (100, token)

        similarity = fuzz.ratio(query.lower(), token.lower())
        if similarity >= threshold:
            return (similarity, token)
        if len(token.split()) > 1:
            for word in token.split():
                simi = fuzz.ratio(
                    query.lower(), word.lower().replace("_", " ")
                )  # word.lower().replace('_',' ')
                if simi >= threshold:
                    return (simi, token)
    return None


def search_compare_similirity_word_load_fulldatabase_to_ram(
    query, database, num_img=10
):
    results = []
    for vid, data in database:
        for k, v in data.items():
            check = find_closest_match(query, v)
            if check is not None:
           
                results.append(
                    {
                        "video_name": vid,
                        "keyframe_id": int(k[:-4]),
                        "score": check[0],
                    }
                )
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    # print(results_sorted[:num_img])
    return results_sorted[:num_img]


def search_in_db_video(vid, data, query):  # tìm trong 1 video
    result = []
    for k, v in data.items():
        check = find_closest_match(query, v)
        if check is not None:
        
            result.append(
                {"video_name": vid, "keyframe_id": int(k[:-4]), "score": check[0]}
            )
    return result


def search_in_db_v2(
    query, database, num_img=10
):  # chạy song song , tìm từng trong từng video. mỗi lần tìm song song trong 14 video
    n_jobs = 28
    results = Parallel(n_jobs=n_jobs)(
        delayed(search_in_db_video)(vid, data, query) for vid, data in database
    )
    results = sum(results, [])  # nối list
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    return results_sorted[:num_img]


class OcrSearcher:
    def __init__(self, ocr_db:OcrDB):
        self.ocr_db = ocr_db

    def search(self, query, num_img):
        return search_in_db_v2(query, self.ocr_db.db, num_img)
