from __future__ import annotations

from typing import Union

import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image

from api.utils import load_model, index_to_url, load_model_open
from open_clip import create_model_from_pretrained, get_tokenizer, create_model_and_transforms




class TextEmbedding():
    def __init__(self,model_name,pretrain):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model,_, self.preprocess = create_model_from_pretrained(model_name, pretrained=pretrain,device=self.device)
        self.model.eval()
    def __call__(self, text: str,model_name) -> np.ndarray:
        tokenizer = open_clip.get_tokenizer(model_name)
        text = tokenizer([text]).to(self.device)


        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text)[0]
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.detach().cpu().numpy()
# use this class if you using a openclip libary model
class searchForOpenClip:
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

        
        self.model,_, self.preprocess = create_model_and_transforms(model_id, pretrained=pretrain,device=self.device)
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
            processed_images = self.processor(images=batch, return_tensors="pt").to(self.device)
            return self.model.get_image_features(**processed_images).detach().cpu().numpy()
            #image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            #with torch.no_grad(), torch.cuda.amp.autocast():
              #image_features = self.model.encode_image(image_input)[0]
              #image_features /= image_features.norm(dim=-1, keepdim=True)
        elif mode=="text":
            # print("batches", batch)
            # text_inputs = open_clip.tokenize(batch).to(self.device)
        
            # with torch.no_grad():
            #     print("text_inputs",text_inputs.shape )
            #     text_feature = self.model.encode_text(text_inputs)
            #     #print(text_feature.shape)
            #     text_feature /= text_feature.norm(dim=-1, keepdim=True)
            
            # return text_feature.detach().cpu().numpy()

            tokenizer = get_tokenizer(self.model_name)
            text = tokenizer(batch).to(self.device)


            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = self.model.encode_text(text)[0]
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


class SemanticSearcher:
    """Object that performs semantic search on images and text

    Parameters
    ----------
    model_id : str
        HuggingFace model id for MultiModal model
    index : faiss.Index, optional
        Faiss index with embeddings to search, by default None
    """

    #def__init__(self, model_id: str, pretrain: str, index: faiss.Index=None, db: list=[])-> None:
        #self.model,_, self.preprocess = open_clip.create_model_and_transforms(self.model_id, pretrained=self.pretrain,device=self.device)
    

    def __init__(self, model_id: str, index: faiss.Index=None, db: list=[]) -> None:
        self.model, self.processor = load_model(model_id)
        self.index = index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            processed_images = self.processor(images=batch, return_tensors="pt").to(self.device)
            return self.model.get_image_features(**processed_images).detach().cpu().numpy()
            #image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            #with torch.no_grad(), torch.cuda.amp.autocast():
              #image_features = self.model.encode_image(image_input)[0]
              #image_features /= image_features.norm(dim=-1, keepdim=True)
        elif mode=="text":
            #text_tokens= open_clip.tokenize().to(device)
            #with torch.no_grad():
            # text_features=  model.encode_text(text_tokens).float()
            #text_features /= text_features.norm(dim=-1, keepdim=True)
            processed_text = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
            return self.model.get_text_features(**processed_text).detach().cpu().numpy()

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

        _, I = self.index.search(query_emb, k)
        I = I.tolist()

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