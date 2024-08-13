import pytest # type: ignore
import torch
from transformers import AutoTokenizer, SqueezeBertModel, SqueezeBertConfig
from Compare_pcc import PCC
import sys
sys.path.append("../")

from reference.SqueezeBert_LayerNorm import SqueezeBertLayerNorm as SqueezeBertLayerNorm_R
from tf_local.models.squeezebert.modeling_squeezebert import SqueezeBertLayerNorm

@pytest.fixture(scope="module")
def base_model():
    configuration = SqueezeBertConfig()
    base_model = SqueezeBertModel(configuration)
    return base_model, configuration

@pytest.mark.parametrize(
    "layer_idx",
    range(12)
)
def test_squeezebert_layernorm_layers(base_model, layer_idx, hidden_size=768):
    
    model, configuration = base_model              
    new_state_dict = PCC.modify_state_dict_with_prefix(model, f'encoder.layers.{layer_idx}.post_attention.layernorm.')

    model = SqueezeBertLayerNorm(hidden_size)
    model_R = SqueezeBertLayerNorm_R(hidden_size)

    model.load_state_dict(new_state_dict)
    model_R.load_state_dict(new_state_dict)

    model.eval()
    model_R.eval()

    input_tensor = torch.rand(1, hidden_size, 8)

    with torch.no_grad():
        output1 = model(input_tensor)
        output2 = model_R(input_tensor)

    output1_flat = PCC.flatten_tuple(output1)
    output2_flat = PCC.flatten_tuple(output2)

    pcc_value = PCC.comp_pcc(output1_flat, output2_flat)

    assert pcc_value[0] > 0.99, f"PCC comparison failed for prefix {layer_idx} with value {pcc_value}"
    print(f"SqueezeBertLayerNorm with prefix {layer_idx} PCC value: {pcc_value}")

if __name__ == "__main__":
    pytest.main()
