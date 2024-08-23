import torch
import numpy as np
from PIL import Image

import open_clip


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
        batch = self._tokenize_texts(texts)

        batch = batch.to(self.device)

        with torch.no_grad():

            text_features = self.model.encode_text(batch)
            if normalization:
                text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()
