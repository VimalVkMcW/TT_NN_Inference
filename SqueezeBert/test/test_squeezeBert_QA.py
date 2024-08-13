import torch
from transformers import AutoTokenizer
import sys
from Compare_pcc import PCC
sys.path.append("../")
from tf_local.models.squeezebert.modeling_squeezebert import SqueezeBertForQuestionAnswering as SqueezeBertForQuestionAnswering_hf
from reference.SqueezeBert_QA import SqueezeBertForQuestionAnswering as SqueezeBertForQuestionAnswering_R

def setup_models():
    tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased", clean_up_tokenization_spaces=False)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    model_hf = SqueezeBertForQuestionAnswering_hf.from_pretrained('squeezebert/squeezebert-uncased')
    model_R = SqueezeBertForQuestionAnswering_R.from_pretrained('/data/SqueezeBert/dataset/')

    new_state_dict = PCC.modify_state_dict_with_prefix(model_hf, '')
    model_hf.load_state_dict(new_state_dict)
    model_R.load_state_dict(new_state_dict)

    return model_hf, model_R, inputs

def test_squeezebert_for_question_answering_outputs():
    model_hf, model_R, inputs = setup_models()

    output_hf = model_hf(**inputs)
    output_R = model_R(**inputs)

    start_logits_hf_flat = PCC.flatten_tuple(output_hf.start_logits)
    start_logits_R_flat = PCC.flatten_tuple(output_R.start_logits)

    end_logits_hf_flat = PCC.flatten_tuple(output_hf.end_logits)
    end_logits_R_flat = PCC.flatten_tuple(output_R.end_logits)

    start_logits_pcc = PCC.comp_pcc(start_logits_hf_flat, start_logits_R_flat)
    end_logits_pcc = PCC.comp_pcc(end_logits_hf_flat, end_logits_R_flat)
    print(start_logits_pcc)
    print(end_logits_pcc)

    assert start_logits_pcc[0] >= 0.99, f"Start Logits PCC is below threshold: {start_logits_pcc}"
    assert end_logits_pcc[0] >= 0.99, f"End Logits PCC is below threshold: {end_logits_pcc}"

if __name__ == "__main__":
    test_squeezebert_for_question_answering_outputs()