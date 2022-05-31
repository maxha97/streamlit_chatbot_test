# KoGPT2-Transformers

KoGPT2 on Huggingface Transformers

### KoGPT2-Transformers

- [SKT-AI 에서 공개한 KoGPT2 (ver 1.0)](https://github.com/SKT-AI/KoGPT2)를 [Transformers](https://github.com/huggingface/transformers)에서 사용하도록 하였습니다.

- **SKT-AI 에서 KoGPT2 2.0을 공개하였습니다.  https://huggingface.co/skt/kogpt2-base-v2/**

### Demo

- 일상 대화 챗봇 : http://demo.tmkor.com:36200/dialo
- 화장품 리뷰 생성 : http://demo.tmkor.com:36200/ctrl

### Example

```python
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")
tokenizer = PreTrainedTokenizerFast.from_pretrained("taeminlee/kogpt2")

input_ids = tokenizer.encode("안녕", add_special_tokens=False, return_tensors="pt")
output_sequences = model.generate(input_ids=input_ids, do_sample=True, max_length=100, num_return_sequences=3)
for generated_sequence in output_sequences:
    generated_sequence = generated_sequence.tolist()
    print("GENERATED SEQUENCE : {0}".format(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))
```