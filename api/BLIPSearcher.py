from __future__ import annotations

from typing import Union

import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image

from api.utils import load_model, index_to_url, load_model_open
from lavis.models import load_model_and_preprocess



# use this class if you using a openclip libary model
class searchForBLIP:
    """Object that performs semantic search on images and text

    Parameters
    ----------
    model_id : str
        open clip model 
    pretrain :str 
        open clip pretrain
    index : faiss.Index, optional
        Faiss index with embeddings to search, by default None
    """
    def __init__(self, model_id: str, pretrain: str , index: faiss.Index=None, db: list=[]) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name= model_id, model_type= pretrain, is_eval=True, device = self.device )
        self.model.eval()
        self.model_name = model_id
        self.index = index
        self.database = db
    

    def process(self, batch: list[Union[Image.Image, str]]) -> np.array:
        """Process a batch of images or text to extract their embeddings

        Parameters
        ----------
        batch : Union[list[Image.Image], list[str]]
            Batch containing images or text

        Returns
        -------
        np.array
            Resulting batch embeddings
        """
        self.model.to(self.device)
        mode = self._infer_type(batch)
        if mode=="visual":
            image_processed = self.vis_processors["eval"](batch).unsqueeze(0).to(self.device)
            sample = {"image": image_processed}
            image_emb = self.model.extract_features(sample, mode="image").image_embeds[0,0,:]
            return image_emb.detach().cpu().numpy()
       
        elif mode=="text":
      

            text_input = self.txt_processors["eval"](batch[0])

            sample = {"text_input": [text_input]}
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features= self.model.extract_features(sample, mode="text").text_embeds[0,0,:] 
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.unsqueeze(0)
            return text_features.detach().cpu().numpy()

    

    def __call__(self, query: list[Union[Image.Image, str]], k: int=5) -> list[str]:
        """Perform a semantic search on a batch of images or text

        Parameters
        ----------
        query : Union[list[Image.Image], list[str]]
            THe input query used to perform the search
        k : int, optional
            Number of items return from query, by default 5

        Returns
        -------
        np.array
            An array containing the indexes of the k most similar items
        """
        # Query embedding
        if not isinstance(query, list):
            query = [query]
        query_emb = self.process(query)
        query_emb /= np.linalg.norm(query_emb)
        # Getting Similarities
        #I: index


        measure = self.index.search(query_emb, k)
        #tuple of two arrays distance , index

        measure  = np.reshape(np.array(measure), newshape=(2, k)).T
        # sort the result
        sorted_indices = np.argsort(measure[:, 0])
        measure  = measure[sorted_indices]

        '''Trả về top K kết quả'''
        search_result = []
        for instance in measure:
            distance, ins_id = instance
            ins_id = int(ins_id)
            video_name, idx = self.database[ins_id][0], self.database[ins_id][1]

            search_result.append({"video_name": video_name,
                                  "keyframe_id": idx,
                                  "score": distance})
        return search_result
    
    @staticmethod
    def _infer_type(x: list[Union[Image.Image, str]]) -> str:
        """Infers the type of the input batch

        Parameters
        ----------
        x : Union[list[Image.Image], list[str]]
            Input batch

        Returns
        -------
        str
            Type of the input batch
        """
        return "visual" if isinstance(x[0], Image.Image) else "text"