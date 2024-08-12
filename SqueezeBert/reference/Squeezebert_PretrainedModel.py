from torch import nn
import sys 
sys.path.append("../")

from utils.modeling_utils import PreTrainedModel
from .SqueezeBert_LayerNorm import SqueezeBertLayerNorm
from .configuration_squeezebert import SqueezeBertConfig 

class SqueezeBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SqueezeBertConfig
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SqueezeBertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
