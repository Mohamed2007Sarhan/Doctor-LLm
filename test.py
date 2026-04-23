from transformers import pipeline

model_path = r"C:\Users\MohamedSarhan\Desktop\hf_models\models--distilgpt2\snapshots\2290a62682d06624634c1f46a6ad5be0f47f38aa"

generator = pipeline("text-generation", model=model_path)

result = generator("Hello", max_length=20)
print(result)