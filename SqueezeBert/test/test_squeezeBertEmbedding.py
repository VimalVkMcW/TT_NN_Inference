import pytest # type: ignore
import torch
from transformers import AutoTokenizer, SqueezeBertModel, SqueezeBertConfig
import sys 
from Compare_pcc import PCC
sys.path.append("../")
from tf_local.models.squeezebert.modeling_squeezebert import SqueezeBertEmbeddings
from reference.SqueezeBertEmbeddings import SqueezeBertEmbeddings as SqueezeBertEmbeddings_R

@pytest.fixture
def setup_models():
    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased",clean_up_tokenization_spaces=False)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    configuration = SqueezeBertConfig()
    base_model = SqueezeBertModel(configuration)

    new_state_dict = PCC.modify_state_dict_with_prefix(base_model, 'embeddings.')

    model = SqueezeBertEmbeddings(configuration)
    model_R = SqueezeBertEmbeddings_R(configuration)

    model.load_state_dict(new_state_dict)
    model_R.load_state_dict(new_state_dict)

    return model, model_R, inputs

def test_squeezebert_embeddings_outputs(setup_models):
    model, model_R, inputs = setup_models

    output1 = model(inputs['input_ids'], inputs['token_type_ids'])
    output2 = model_R(inputs['input_ids'], inputs['token_type_ids'])

    output1_flat = PCC.flatten_tuple(output1)
    output2_flat = PCC.flatten_tuple(output2)

    pcc_value = PCC.comp_pcc(output1_flat, output2_flat)

    assert pcc_value[0] >= 0.99, f"PCC is below threshold: {pcc_value}"

if __name__ == "__main__":
    pytest.main()
