"""Contains encoders and decoders"""

from configuration import GPT2Config

encoders = {}


from modeling.gpt2 import MyGPT2Decoder

decoders = {
    'gpt2': (GPT2Config, MyGPT2Decoder),
}

