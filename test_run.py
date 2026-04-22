from tools.repair_tools import _test_model_loading
import json

path = r"C:\Users\MohamedSarhan\Desktop\hf_models\models--distilgpt2\snapshots\2290a62682d06624634c1f46a6ad5be0f47f38aa"
result = _test_model_loading(path, "cpu")
print(result.output)
