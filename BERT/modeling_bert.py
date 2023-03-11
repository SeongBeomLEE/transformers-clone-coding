import torch.nn as nn
from .config_bert import BertConfig
from base.model import BaseModel

class BertModel(BaseModel):
    config_class = BertConfig

    def __init__(self, config, add_pooling_layer=True) -> None:
        super().__init__(config)
        self.config = config
        self.fc = nn.Linear(1, 10)
        # TODO : self.embedding = BertEmbeddings(config)
        # TODO : self.encoder = BertEncoder(config)
        # TODO : self.pooler = BertPooler(config) if add_pooling_layer else None
        # Initialize weights and apply final processing
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, a):
        a = self.fc(a)
        return a