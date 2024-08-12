import pytest # type: ignore
import torch
import sys
from Compare_pcc import PCC
from transformers import SqueezeBertConfig, SqueezeBertModel
sys.path.append("../")
from tf_local.models.squeezebert.modeling_squeezebert import SqueezeBertModule
from reference.SqueezeBert_Module import SqueezeBertModule as SqueezeBertModule_R

@pytest.fixture(scope="module")
def base_model():
    configuration = SqueezeBertConfig()
    base_model = SqueezeBertModel(configuration)
    return base_model, configuration

@pytest.mark.parametrize(
    "layer_idx",
    range(12)
)
def test_squeezebert_module_layers(base_model, layer_idx):
    model, configuration = base_model

    # Modify state dict for the specific layer
    new_state_dict = PCC.modify_state_dict_with_prefix(model, f'encoder.layers.{layer_idx}.')

    torch_model = SqueezeBertModule(configuration)
    ref_model = SqueezeBertModule_R(configuration)

    torch_model.load_state_dict(new_state_dict)
    ref_model.load_state_dict(new_state_dict)

    torch_model.eval()
    ref_model.eval()

    hidden_states = torch.rand(1, 768, 8)
    attention_mask = -torch.zeros(1, 1, 1, 8)

    with torch.no_grad():
        torch_output = torch_model(hidden_states, attention_mask, False)
        ref_output = ref_model(hidden_states, attention_mask, False)

    output1_flat = PCC.flatten_tuple(torch_output['feature_map'])
    output2_flat = PCC.flatten_tuple(ref_output['feature_map'])

    result, pcc_value = PCC.comp_pcc(output1_flat, output2_flat)
    assert result, f"SqueezeBert module with prefix {layer_idx} does not match. PCC value: {pcc_value}"
    print(f"SqueezeBert module with prefix {layer_idx} PCC value: {pcc_value}")

if __name__ == "__main__":
    pytest.main()
