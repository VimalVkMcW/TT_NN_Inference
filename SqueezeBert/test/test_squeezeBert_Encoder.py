import pytest # type: ignore
import torch
from transformers import AutoTokenizer, SqueezeBertModel, SqueezeBertConfig
from Compare_pcc import PCC
import sys
sys.path.append("../")

from reference.SqueezeBert_Encoder import SqueezeBertEncoder as SqueezeBertEncoder_R
from tf_local.models.squeezebert.modeling_squeezebert import SqueezeBertEncoder

@pytest.fixture(scope="module")
def setup_squeezebert_encoder_models():

    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased", clean_up_tokenization_spaces=False)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    
    configuration = SqueezeBertConfig()
    base_model = SqueezeBertModel(configuration)
    new_state_dict = PCC.modify_state_dict_with_prefix(base_model, 'encoder.')

    model = SqueezeBertEncoder(configuration)
    model_R = SqueezeBertEncoder_R(configuration)

    model.load_state_dict(new_state_dict)
    model_R.load_state_dict(new_state_dict)

    return model, model_R, inputs

@pytest.mark.parametrize(
    "prefix, hidden_size, seq_length",
    [
        ('encoder', 768, 8), 
    ]
)
def test_squeezebert_encoder_output(setup_squeezebert_encoder_models, prefix, hidden_size, seq_length):
    model, model_R, inputs = setup_squeezebert_encoder_models
    hidden_states = torch.rand(1, seq_length, hidden_size)
    attention_mask = -torch.zeros(1, 1, 1, seq_length)

    with torch.no_grad():
        output1 = model(hidden_states, attention_mask, None, False, True)
        output2 = model_R(hidden_states, attention_mask, None, False, True)

    hidden_states_flat = PCC.flatten_tuple(output1[0])
    hidden_states_R_flat = PCC.flatten_tuple(output2[0])
    attention_mask_flat = PCC.flatten_tuple(output1[1])
    attention_mask_R_flat = PCC.flatten_tuple(output2[1])

    pcc_hidden_states = PCC.comp_pcc(hidden_states_flat, hidden_states_R_flat)
    pcc_attention_mask = PCC.comp_pcc(attention_mask_flat, attention_mask_R_flat)

    assert pcc_hidden_states[0] > 0.99, f"PCC comparison failed for hidden states with value {pcc_hidden_states}"
    assert pcc_attention_mask[0] > 0.99, f"PCC comparison failed for attention mask with value {pcc_attention_mask}"
    
    print(f"PCC Value for hidden states: {pcc_hidden_states}")
    print(f"PCC Value for attention mask: {pcc_attention_mask}")

if __name__ == "__main__":
    pytest.main()
