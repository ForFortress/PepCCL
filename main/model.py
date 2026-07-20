import re, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5Model,T5EncoderModel
from collections.abc import Iterable

# GPU
device = torch.device("cuda:0")
t5_path_path = "PRE-TRAINED T5 MODEL PATH"

def freeze_unfreeze_layers(model, layer_indexs, unfreeze=False):
    if type(layer_indexs) == int:
        layer_idx = layer_indexs
        freeze_layer = model.block[layer_idx]
        for para in freeze_layer.parameters():
            para.requires_grad_(unfreeze)
    else:
        start = layer_indexs[0]
        end = layer_indexs[1]
        freeze_layers = model.block[start: end+1]
        for la in freeze_layers:
            for para in la.parameters():
                para.requires_grad_(unfreeze)
        freeze_layer = model.final_layer_norm
        for para in freeze_layer.parameters():
            para.requires_grad_(unfreeze)

def freeze(self): 
    for name, child in self.t5.named_children():
        for param in child.parameters():
            param.requires_grad = False
    freeze_unfreeze_layers(self.t5.encoder, (20, 23), unfreeze=True) # unfreeze the last 1/6 encoder blocks of the T5 encoder
    
    for name, param in self.t5.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")

class t5(nn.Module):
    def __init__(self, config):
        super(t5, self).__init__()
        global max_len, d_model, device
        d_model = 1024

        t5_path = t5_path_path
        self.tokenizer = T5Tokenizer.from_pretrained(t5_path, do_lower_case=False)
        self.t5 = T5EncoderModel.from_pretrained(t5_path)

        freeze(self)

        self.block1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.block2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )

    def forward(self, input_seq):
        input_seq = ' '.join(input_seq)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        encoded_input = self.tokenizer(input_seq, return_tensors='pt')
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].cuda(device)
        output = self.t5(**encoded_input)
        output = output[0]
        representation = output.view(-1, 1024)
        representation = representation[0:-1]
        representation = self.block1(representation)
        return representation
    
    def get_logits(self, x):
        with torch.no_grad():
            output = self.forward(x)
        logits = self.block2(output)
        return logits
