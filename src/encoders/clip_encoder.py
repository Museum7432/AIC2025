import torch
import numpy as np
from PIL import Image

import open_clip

from collections import OrderedDict

class LimitedCache:
    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key not in self.cache:
            return None
        # Move the accessed item to the end to show that it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)  # If the key exists, move it to the end
        self.cache[key] = value
        # If the cache exceeds the maximum size, remove the oldest item
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def have(self, key):
        return key in self.cache

    def clear(self):
        self.cache.clear()

class ClipEncoder:
    def __init__(self, model_arch, pretrained, device=None, jit=True):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # load model
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_arch, pretrained=pretrained, device=self.device, jit=jit
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        self.img_preprocess = preprocess

        self.text_tokenizer = open_clip.get_tokenizer(model_arch)

        # TODO: use service streamer to handle large amount of requests

        self.cache = LimitedCache(1000)

    def _tokenize_texts(self, texts):
        # open clip has a token limit of 77
        # (#sents, 77)
        return self.text_tokenizer(texts)

    def _preprocess_images(self, images):
        # images should be a list of PIL.Image
        # TODO: optimize this
        return torch.vstack([self.img_preprocess(i).unsqueeze(0) for i in images])

    def encode_images(self, images, normalization=True):
        # images should be a list of PIL.Image

        if len(images) == 0:
            return []

        batch = self._preprocess_images(images)

        batch = batch.to(self.device)

        with torch.no_grad():

            image_features = self.model.encode_image(batch)

            if normalization:
                image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()

    def encode_texts(self, texts, normalization=True):

        if len(texts) == 0:
            return []
        

        cache_hits = [i for i, t in enumerate(texts) if self.cache.have(t)]

        cache_features = [self.cache.get(texts[i]) for i in cache_hits]
        
        uncached_ids = [i for i, t in enumerate(texts) if i not in cache_hits]

        uncached_texts = [texts[i] for i in uncached_ids]

        result = [None for _ in range(len(texts))]

        for i, f in zip(cache_hits ,cache_features):
            result[i] = f

        if len(uncached_ids) > 0:

            batch = self._tokenize_texts(uncached_texts)

            batch = batch.to(self.device)

            with torch.no_grad():

                text_features = self.model.encode_text(batch)
                if normalization:
                    text_features /= text_features.norm(dim=-1, keepdim=True)

            uncached_features = text_features.cpu().numpy()

            
            for i, t, f in zip(uncached_ids, uncached_texts, uncached_features):
                self.cache.set(t, f)

                result[i] = f

        return np.vstack(result)
