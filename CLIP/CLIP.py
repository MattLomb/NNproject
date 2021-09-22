import tensorflow as tf
import numpy as np
from CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class CLIP:
    def __init__(self):
        self.__load_models()

    def __load_models(self):
        self.model = tf.keras.models.load_model("CLIP/model")
        self.model.compile()

    '''
    Incompatible shapes: [50,768] vs. [1,1025,768]
    PyTorch
    _transform(model.input_resolution.item())
    
    def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    '''

    def predict(self, img_input, text_input):
        return self.model.predict((img_input, text_input))

    def tokenize(self, texts, context_length: int = 77,
                 truncate: bool = False):
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP-tf2 models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
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

    def get_model(self):
        return self.model

