import tensorflow as tf
import tensorflow_text  # Needed for models
import numpy as np
from CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class CLIP:
    def __init__(self):
        self.__load_models()

    def __load_models(self):
        # self.model = tf.keras.models.load_model("CLIP/models/general")
        self.text_model = tf.keras.models.load_model("CLIP/models/text_encoder")
        self.image_model = tf.keras.models.load_model("CLIP/models/vision_encoder")

    def predict_text(self, text_input):
        return self.text_model.predict(text_input)

    def predict_image(self, image_input):
        return self.image_model.predict(image_input)

    def tokenize(self, texts, context_length: int = 77,
                 truncate: bool = False):

        if isinstance(texts, str):
            texts = [texts]

        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
        # result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        result = np.zeros((len(all_tokens), context_length))

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = tokens

        return result

    def get_text_model(self):
        return self.text_model

    def get_image_model(self):
        return self.image_model
