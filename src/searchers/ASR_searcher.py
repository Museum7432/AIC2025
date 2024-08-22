from fuzzywuzzy import process, fuzz
import time
from joblib import Parallel, delayed
import pandas as pd


# PATH_TO_MAP_KEYFRAMES_FOLDER="./map-keyframes/"
def find_closest_match_fast(query, sentence, threshold=55):
    if query.lower() in sentence["text"].lower():
        return (100, sentence)
    # score = fuzz.ratio(query.lower(), sentence["text"].lower())
    # if score >= threshold:
    #         return (score, sentence)
    return None


# def get_file_name_img_from_keyframe(vid,lst_frame_idx):
#     df=pd.read_csv(PATH_TO_MAP_KEYFRAMES_FOLDER+vid+".csv")
#     lst=[]
#     for i in range(len(lst_frame_idx)):
#         lst_dis=[]
#         #chỗ này chỉnh lại binary search cho nhanh vì mảng được sắp xếp r.
#         flag=0
#         for j,frame_idx in enumerate(df.frame_idx):
#             if lst_frame_idx[i]-frame_idx<0:
#               flag+=1
#               if flag==2:
#                 break
#             distance=abs(frame_idx-lst_frame_idx[i])
#             lst_dis.append(distance)
#         lst.append(df.n[lst_dis.index(min(lst_dis))])
#     return lst


# def get_file_name_img_from_keyframe(vid,lst_frame_idx):
#     df=pd.read_csv(PATH_TO_MAP_KEYFRAMES_FOLDER+vid+".csv")
#     lst=[]
#     for i in range(len(lst_frame_idx)):
#         for j,frame_idx in enumerate(df.frame_idx):
#             if frame_idx==lst_frame_idx[i]:
#                 lst.append(df.n[j])
#                 break
#     return lst
def search_in_db_video_fast(vid, data, query):  # tìm trong 1 video
    result = []
    for idx, sentence in enumerate(data):  # duyệt qua từng câu nói trong video
        check = find_closest_match_fast(query, sentence)
        if check is not None:
            # img_names=get_file_name_img_from_keyframe(vid,check[1]['frames'])
            # result.append({"video_name": vid, "keyframe_id": img_names, "score": check[0]})
            result.append(
                {
                    "video_name": vid,
                    "keyframe_id": check[1]["frames"],
                    "score": check[0],
                    "text": check[1]["text"],
                }
            )

    return result


def ASR_search_engine_fast(
    query, database, num_img=10
):  # chạy song song , tìm từng trong từng video. mỗi lần tìm song song trong 14 video
    n_jobs = 20
    query = query.strip()
    results = Parallel(n_jobs=n_jobs)(
        delayed(search_in_db_video_fast)(vid, data, query) for vid, data in database
    )
    results = sum(results, [])  # nối list
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    return results_sorted[:num_img]


def find_closest_match_slow(query, sentence, threshold=55):
    if query.lower() in sentence["text"].lower():
        return (100, sentence)
    score = fuzz.ratio(query.lower(), sentence["text"].lower())
    if score >= threshold:
        return (score, sentence)
    return None


def search_in_db_video_slow(vid, data, query):  # tìm trong 1 video
    result = []
    for idx, sentence in enumerate(data):  # duyệt qua từng câu nói trong video
        check = find_closest_match_slow(query, sentence)
        if check is not None:
            result.append(
                {
                    "video_name": vid,
                    "keyframe_id": check[1]["frames"],
                    "score": check[0],
                    "text": check[1]["text"],
                }
            )

    return result


def ASR_search_engine_slow(
    query, database, num_img=10
):  # chạy song song , tìm từng trong từng video. mỗi lần tìm song song trong 14 video
    n_jobs = 20
    query = query.strip()
    query = query.strip()
    results = Parallel(n_jobs=n_jobs)(
        delayed(search_in_db_video_slow)(vid, data, query) for vid, data in database
    )
    results = sum(results, [])  # nối list
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    return results_sorted[:num_img]


class AsrSearcher:
    def __init__(self, asr_db):
        self.asr_db = asr_db

    def search_fast(self, text, num_img):
        return ASR_search_engine_fast(text, self.asr_db.db, num_img=num_img)

    def search_slow(self, text, num_img):
        return ASR_search_engine_slow(text, self.asr_db.db, num_img=num_img)
