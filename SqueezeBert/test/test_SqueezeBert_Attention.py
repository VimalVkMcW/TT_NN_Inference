import pytest # type: ignore
from transformers import AutoTokenizer , SqueezeBertModel
import torch
import sys 
from Compare_pcc import PCC
sys.path.append("../")

from reference.SqueezeBert_Attention import SqueezeBertSelfAttention as SqueezeBertSelfAttention_R
from tf_local.models.squeezebert.modeling_squeezebert import SqueezeBertSelfAttention

from transformers import SqueezeBertConfig
@pytest.fixture(scope="module")
def squeezebert_model():
    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased", clean_up_tokenization_spaces=False)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    configuration = SqueezeBertConfig()
    model = SqueezeBertModel(configuration)

    return model, configuration, inputs

@pytest.mark.parametrize("layer_idx", range(12))  
def test_conv_dropout_layernorm_layers(layer_idx, squeezebert_model):
    model, configuration, inputs = squeezebert_model

    new_state_dict = PCC.modify_state_dict_with_prefix(model, f'encoder.layers.{layer_idx}.attention.')

    model = SqueezeBertSelfAttention(configuration,768,4,4,4)
    model_R = SqueezeBertSelfAttention_R(configuration,768,4,4,4)

    model.load_state_dict(new_state_dict)
    model_R.load_state_dict(new_state_dict)
    
    hidden_states = torch.rand(1,768,8)
    attention_mask = -torch.zeros(1, 1, 1, 8)
    

    with torch.no_grad():
        output1 = model(hidden_states, attention_mask, False)
        output2 = model_R(hidden_states, attention_mask, False)

    output1_flat = PCC.flatten_tuple(output1['context_layer'])
    output2_flat = PCC.flatten_tuple(output2['context_layer'])

    pcc_value = PCC.comp_pcc(output1_flat, output2_flat)

    assert pcc_value[0] > 0.99, f"PCC comparison failed for layer {layer_idx} with value {pcc_value}"
    print(f"ConvDropoutLayerNorm layer {layer_idx} PCC value: {pcc_value}")

if __name__ == "__main__":
    pytest.main()