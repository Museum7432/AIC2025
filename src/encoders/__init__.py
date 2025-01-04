from .clip_encoder import ClipEncoder
from .blip_encoder import BlipEncoder

# from collections import OrderedDict

# class LimitedCache:
#     def __init__(self, max_size):
#         self.cache = OrderedDict()
#         self.max_size = max_size

#     def get(self, key):
#         if key not in self.cache:
#             return None
#         # Move the accessed item to the end to show that it was recently used
#         self.cache.move_to_end(key)
#         return self.cache[key]

#     def set(self, key, value):
#         if key in self.cache:
#             self.cache.move_to_end(key)  # If the key exists, move it to the end
#         self.cache[key] = value
#         # If the cache exceeds the maximum size, remove the oldest item
#         if len(self.cache) > self.max_size:
#             self.cache.popitem(last=False)

#     def clear(self):
#         self.cache.clear()

# class UnifiedEncoder:
#     def __init__(self, model_arch, pretrained, is_clip_model=True, device=None, jit=True):
        
#         if is_clip_model:
#             self.model = ClipEncoder(model_arch, pretrain_name, device=device, jit=jit)
#         else:
#             self.model = BlipEncoder(model_arch, pretrain_name, device=device)

#         self.text_cache
#     def encode_texts(self, texts, normalization=True):
#         if len(texts) == 0:
#             return np.array([])
        

            
