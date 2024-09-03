import math
import numpy as np
from database import ObjectCountDB

def search_obj_count_engine_slow(query:str,
                  db: list,
                  topk:int=10,
                  measure_method: str="l2_norm"):
      # handle query
      lst=query.split()
      class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
                        'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
                        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
                        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                        'hair drier', 'toothbrush']
      
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
def search_obj_count_engine_fast(query:str,  topk:int , 
                  db: list,

                  ):
  # handle query
  lst=query.split()

  # fmt: off
  class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                     'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
                     'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
                     'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
                     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                     'hair drier', 'toothbrush']
  # fmt: on

  class_dict = {name: i for i, name in enumerate(class_names)}

  lst_set=[]
  for num_cls in lst:
        num,cls=num_cls.split('-')
        set_img_of_numobj_cls=db[class_dict[cls]][int(num)]
        lst_set.append(set_img_of_numobj_cls)
        
  result = lst_set[0]
  for s in lst_set[1:]:
      result = result.intersection(s)
  search_result = []

  k =0 
  for vid_img in result:
     k+=1
     if k == topk: 
       break
     vid,img=vid_img
     search_result.append({"video_name":vid,
                          "keyframe_id": img,
                          "score": 0})
  return search_result

class ObjectCountSearcher:
  def __init__(self, obj_db: ObjectCountDB):
    self.db = obj_db
  
  def search_fast(self, query, topk=5):
    return search_obj_count_engine_fast(query, topk , self.db.fast_db)
  
  def search_slow(self, query, topk=5, measure_method="l2_norm"):
    return search_obj_count_engine_slow(query, self.db.slow_db, topk=topk, measure_method=measure_method)
