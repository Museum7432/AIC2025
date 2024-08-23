import torch
import numpy as np
from PIL import Image

from lavis.models import load_model_and_preprocess


class BlipEncoder:
    def __init__(self, model_id, pretrain, device=None):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # load model
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name=model_id, model_type=pretrain, is_eval=True, device=self.device
        )

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        self.img_preprocess = vis_processors["eval"]

        self.text_preprocess = txt_processors["eval"]

    def _preprocess_texts(self, texts):
        return [self.text_preprocess(t) for t in texts]

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

        sample = {"image": batch}

        with torch.no_grad():

            image_features = self.model.extract_features(
                sample, mode="image"
            ).image_embeds[:, 0, :]

            if normalization:
                image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()

    def encode_texts(self, texts, normalization=True):
        if len(texts) == 0:
            return []
        texts = self._preprocess_texts(texts)

        sample = {"text_input": texts}

        with torch.no_grad():

            text_features = self.model.extract_features(
                sample, mode="text"
            ).text_embeds[:, 0, :]
            if normalization:
                text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()
