import math
import numpy as np
def search_obj_count_engine_slow(query:str,
                  db: list,
                  topk:int=10,
                  measure_method: str="l2_norm"):
  # handle query
  lst=query.split()
  class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

  class_dict = {name: i for i, name in enumerate(class_names)}
  dic_img={}
  for key in class_dict.keys():
        dic_img[key]=0
  for num_cls in lst:
        num,cls=num_cls.split('-')
        dic_img[cls]=int(num)
  query_arr=np.array(list(dic_img.values()))
  indices=[]
  for i,key in enumerate(dic_img):
      if dic_img[key]!=0:
          indices.append(i)
  query_arr=query_arr[indices]


  # '''Duyệt tuyến tính và tính độ tương đồng giữa 2 vector'''
  measure = []
  for ins_id, instance in enumerate(db):
    video_name, idx, feat_arr= instance
    #để nhanh hơn thì mấy thằng(img) mà ko chứa đủ class trong query cho cút
    feat_arr=feat_arr[indices]
    if np.any((feat_arr==0)):
      continue
    if measure_method=="l1_norm":
      distance = -1 * np.mean([abs(q - t) for q, t in zip(query_arr, feat_arr)])# thêm dấu - để nó đảo chiều cùng mấy độ đo kia
    elif measure_method=="l2_norm":
      distance= -1*np.sqrt(((feat_arr-query_arr)**2).sum())
    measure.append((ins_id, distance))

  '''Sắp xếp kết quả'''

  measure = sorted(measure, key=lambda x:x[-1], reverse=True)

  '''Trả về top K kết quả'''
  search_result = []
  for instance in measure[:topk]:
    ins_id, distance = instance
    video_name, idx, _ = db[ins_id]
    search_result.append({"video_name":video_name,
                          "keyframe_id": idx,
                          "score": distance})
  return search_result
def search_obj_count_engine_fast(query:str,
                  db: list,
                  ):
   pass