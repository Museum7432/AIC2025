import numpy as np
from database import ObjectLocationDB


class ObjectLocationSearcher:
    def __init__(self, obj_loc_db: ObjectLocationDB):
        self.db = obj_loc_db

        self.elastic_client = obj_loc_db.elastic_client

    def elastic_search(self, class_ids, box_cords, topk):

        assert len(class_ids) == len(box_cords)

        query = [
            self.db.enocde_pos(class_idx, box_cordinate)
            for class_idx, box_cordinate in zip(class_ids, box_cords)
        ]

        query = " ".join(query)

        print(query)

        es_query = {
            "bool": {
                "should": [
                    {
                        "match": {
                            "text": {
                                "query": query,
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "fuzziness": 2,
                                "prefix_length": self.db.prefix_len,  # important, or else it will
                                # perfom fuzzy search on the class name
                                "fuzzy_transpositions": False,
                                # "boost": 0.5,
                            }
                        },
                    },
                ]
            }
        }

        hits = self.elastic_client.search(
            index="color_obj_loc",
            query=es_query,
            size=topk,
        ).raw["hits"]["hits"]

        results = [
            {
                "video_name": d["_source"]["vid_name"],
                "keyframe_id": d["_source"]["keyframe_id"],
                # "score": score,
                "score": d["_score"],
                "text": d["_source"]["text"],
            }
            for d in hits
        ]

        return results
