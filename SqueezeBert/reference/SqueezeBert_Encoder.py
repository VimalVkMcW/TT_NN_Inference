from torch import nn
from .SqueezeBert_Module import SqueezeBertModule
import sys 
sys.path.append('../')
from utils.modeling_outputs import BaseModelOutput
class SqueezeBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embedding_size == config.hidden_size, (
            "If you want embedding_size != intermediate hidden_size, "
            "please insert a Conv1d layer to adjust the number of channels "
            "before the first SqueezeBertModule."
        )

        self.layers = nn.ModuleList(SqueezeBertModule(config) for _ in range(config.num_hidden_layers))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if head_mask is None:
            head_mask_is_all_none = True
        elif head_mask.count(None) == len(head_mask):
            head_mask_is_all_none = True
        else:
            head_mask_is_all_none = False
        assert head_mask_is_all_none is True, "head_mask is not yet supported in the SqueezeBert implementation."

        # [batch_size, sequence_length, hidden_size] --> [batch_size, hidden_size, sequence_length]
        hidden_states = hidden_states.permute(0, 2, 1)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                hidden_states = hidden_states.permute(0, 2, 1)
                all_hidden_states += (hidden_states,)
                hidden_states = hidden_states.permute(0, 2, 1)

            layer_output = layer.forward(hidden_states, attention_mask, output_attentions)

            hidden_states = layer_output["feature_map"]

            if output_attentions:
                all_attentions += (layer_output["attention_score"],)

        # [batch_size, hidden_size, sequence_length] --> [batch_size, sequence_length, hidden_size]
        hidden_states = hidden_states.permute(0, 2, 1)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )