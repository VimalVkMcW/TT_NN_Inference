import pytest # type: ignore
import torch
from transformers import AutoTokenizer, SqueezeBertModel, SqueezeBertConfig
from Compare_pcc import PCC
import sys

sys.path.append("../")

from reference.SqueezeBert_ConvDropoutLayerNorm import ConvDropoutLayerNorm as ConvDropoutLayerNorm_R
from tf_local.models.squeezebert.modeling_squeezebert import ConvDropoutLayerNorm

@pytest.fixture(scope="module")
def squeezebert_model():
    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased", clean_up_tokenization_spaces=False)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    configuration = SqueezeBertConfig()
    model = SqueezeBertModel(configuration)

    return model, inputs

@pytest.mark.parametrize("layer_idx", range(12))  
def test_conv_dropout_layernorm_layers(layer_idx, squeezebert_model):
    model, inputs = squeezebert_model

    new_state_dict = PCC.modify_state_dict_with_prefix(model, f'encoder.layers.{layer_idx}.post_attention.')

    model = ConvDropoutLayerNorm(768, 768, 1, 0)
    model_R = ConvDropoutLayerNorm_R(768, 768, 1, 0)

    model.load_state_dict(new_state_dict)
    model_R.load_state_dict(new_state_dict)

    hidden_states = torch.rand(1, 768, 8)
    input_tensor = torch.rand(1, 768, 8)

    with torch.no_grad():
        output1 = model(hidden_states, input_tensor)
        output2 = model_R(hidden_states, input_tensor)

    output1_flat = PCC.flatten_tuple(output1)
    output2_flat = PCC.flatten_tuple(output2)

    pcc_value = PCC.comp_pcc(output1_flat, output2_flat)

    assert pcc_value[0] > 0.99, f"PCC comparison failed for layer {layer_idx} with value {pcc_value}"
    print(f"ConvDropoutLayerNorm layer {layer_idx} PCC value: {pcc_value}")

if __name__ == "__main__":
    pytest.main()
