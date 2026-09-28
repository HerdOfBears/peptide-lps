"""
Wrapper for fetching ESM protein language model. 
"""

import os
import torch
import pandas as pd
from transformers import AutoModel,AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast
from sklearn.decomposition import PCA

class Encoder():

    def __init__(self, model_name_or_path="ESMC-300M"):

        if not os.path.exists(model_name_or_path):
            _to_load = f"biohub/{model_name_or_path}"
        else:
            _to_load = model_name_or_path
        
        # model_name = "ESMC-300M" # or -600M, -6B
        self.model = AutoModelForMaskedLM.from_pretrained(_to_load, device_map="auto", trust_remote_code=True).eval()

        if "ESMC-" in _to_load:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(_to_load)
            self.tokenizer.model_input_names = ["input_ids", "attention_mask"]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(   _to_load,trust_remote_code=True)

    def encode(self, sequence:str|list[str]) -> torch.tensor:
        """
        Encode a sequence or list of sequences into the representation space 
        of the model.

        Parameters:
        -----------
        sequence: str or list of str
            The protein or batch of proteins.
            Alphabet must correspond to alphabet of the encoding model.
        
        Returns:
        --------
        representations: torch.tensor
            The encoded sequence representation.
        """

        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model(**inputs)

        print(output.keys())
        return output.hidden_states[-1]

class EncoderWithPCA():
    """
    Put a PCA 'head' where we take the top n_pc_components principal components.
    """
    def __init__(self, train_set_path:str, n_pc_components:int=5, model_name_or_path:str="ESMC-300M"):

        self.encoder = Encoder(model_name_or_path)

        self.PCA = PCA(n_pc_components)

        # load the training set
        train_set = pd.read_csv(train_set_path)
        _sequences = train_set["sequence"].to_list()

        # embed them
        _encodings = self.encoder.encode(_sequences) # len(sequences), d_latent

        # PCA it all
        self.PCA.fit(_encodings)

    def encode(self, sequence:str|list[str])->torch.tensor:

        _encoding    = self.encoder.encode(sequence)
        _pc_encoding = self.PCA.transform(_encoding)

        return _pc_encoding