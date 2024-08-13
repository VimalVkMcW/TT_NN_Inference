import pytest # type: ignore
import torch
from transformers import AutoTokenizer, SqueezeBertModel, SqueezeBertConfig
from Compare_pcc import PCC
import sys
sys.path.append("../")

from reference.SqueezeBert_Matmul import MatMulWrapper as MatMulWrapper_R
from tf_local.models.squeezebert.modeling_squeezebert import MatMulWrapper

@pytest.fixture
def setup_matmul_wrapper_models():
    model = MatMulWrapper()
    model_R = MatMulWrapper_R()

    return model, model_R

def test_matmul_wrapper_output(setup_matmul_wrapper_models):
    model, model_R = setup_matmul_wrapper_models
    a = torch.rand(1, 12, 8, 8)
    b = torch.rand(1, 12, 8, 64)

    output1 = model(a, b)
    output2 = model_R(a, b)

    output1_flat = PCC.flatten_tuple(output1)
    output2_flat = PCC.flatten_tuple(output2)

    pcc_value = PCC.comp_pcc(output1_flat, output2_flat)

    assert pcc_value[0] > 0.99, f"PCC comparison failed with value {pcc_value}"

if __name__ == "__main__":
    pytest.main()
