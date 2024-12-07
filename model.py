import torch
import torch.nn as nn
from transformers import AutoModel

class Net(nn.Module):
    def __init__(self, device='cpu', hidden_size=None, finetuning=False, tag_size=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(hidden_size, tag_size)

        self.device = device
        self.finetuning = finetuning

    def forward(self, x, y, ):
        x = x.to(self.device)
        y = y.to(self.device)

        if self.finetuning:
            self.bert.train()
            encoded_layers = self.bert(x)
            enc = encoded_layers[0]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers = self.bert(x)
                enc = encoded_layers[0]
        
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)

        return logits, y, y_hat
