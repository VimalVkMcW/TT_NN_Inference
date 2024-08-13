from transformers import AutoTokenizer
import sys 
sys.path.append("../")
# from reference.SqueezeBert import SqueezeBertModel
from tf_local.models.squeezebert.modeling_squeezebert import SqueezeBertModel


tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased",clean_up_tokenization_spaces=False)
model = SqueezeBertModel.from_pretrained("squeezebert/squeezebert-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# tokens = tokenizer(inputs)
# print(inputs)

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)

# hidden_states torch.Size([1, 768, 8])
# input_tensor torch.Size([1, 768, 8])





# (self, cin, cout, groups, dropout_prob)
# 768 768 1 0.1