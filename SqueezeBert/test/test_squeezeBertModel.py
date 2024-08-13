import pytest # type: ignore
import torch
from transformers import AutoTokenizer
import sys 
from Compare_pcc import PCC
sys.path.append("../")
from tf_local.models.squeezebert.modeling_squeezebert import SqueezeBertModel as SqueezeBertModel_hf
from reference.SqueezeBert_Model import SqueezeBertModel as SqueezeBertModel_R

@pytest.fixture
def setup_models():
    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased",clean_up_tokenization_spaces=False)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    model_hf = SqueezeBertModel_hf.from_pretrained('squeezebert/squeezebert-uncased')
    model_R = SqueezeBertModel_R.from_pretrained('/data/SqueezeBert/dataset/')

    new_state_dict = PCC.modify_state_dict_with_prefix(model_hf, '')
    
    model_hf.load_state_dict(new_state_dict)
    model_R.load_state_dict(new_state_dict)

    return model_hf, model_R, inputs

def test_squeezebert_model_outputs(setup_models):
    model_hf, model_R, inputs = setup_models

    output_hf = model_hf(**inputs)
    output_R = model_R(**inputs)

    last_hidden_state1_flat = PCC.flatten_tuple(output_hf[0])
    last_hidden_state2_flat = PCC.flatten_tuple(output_R[0])

    pooler_output1_flat = PCC.flatten_tuple(output_hf[1])
    pooler_output2_flat = PCC.flatten_tuple(output_R[1])

    last_hidden_state_pcc = PCC.comp_pcc(last_hidden_state1_flat, last_hidden_state2_flat)
    pooler_output_pcc = PCC.comp_pcc(pooler_output1_flat, pooler_output2_flat)

    assert last_hidden_state_pcc[0] >= 0.99, f"Last Hidden State PCC is below threshold: {last_hidden_state_pcc}"
    assert pooler_output_pcc[0] >= 0.99, f"Pooler Output PCC is below threshold: {pooler_output_pcc}"

if __name__ == "__main__":
    pytest.main()
