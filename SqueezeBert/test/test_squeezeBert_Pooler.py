import pytest # type: ignore
import torch
from transformers import AutoTokenizer, SqueezeBertConfig, SqueezeBertModel
import sys 
from Compare_pcc import PCC
sys.path.append("../")
from tf_local.models.squeezebert.modeling_squeezebert import SqueezeBertPooler
from reference.SqueezeBert_Pooler import SqueezeBertPooler as SqueezeBertPooler_R

@pytest.fixture
def setup_models():
    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased",clean_up_tokenization_spaces=False)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    configuration = SqueezeBertConfig()
    base_model = SqueezeBertModel(configuration)

    new_state_dict = PCC.modify_state_dict_with_prefix(base_model, 'pooler.')

    model = SqueezeBertPooler(configuration)
    model_R = SqueezeBertPooler_R(configuration)

    model.load_state_dict(new_state_dict)
    model_R.load_state_dict(new_state_dict)

    return model, model_R

def test_squeezebert_pooler_outputs(setup_models):
    model, model_R = setup_models

    hidden_states = torch.rand(1, 8, 768)

    output1 = model(hidden_states)
    output2 = model_R(hidden_states)

    output1_flat = PCC.flatten_tuple(output1)
    output2_flat = PCC.flatten_tuple(output2)

    pcc_value = PCC.comp_pcc(output1_flat, output2_flat)

    assert pcc_value[0] >= 0.99, f"PCC is below threshold: {pcc_value}"

if __name__ == "__main__":
    pytest.main()
