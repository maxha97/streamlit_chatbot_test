

import streamlit as st
import torch

from tokenizers import SentencePieceBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel

def encoding(text, tokenizer):
    text = '<s>'+text+'</s><s>'
    return torch.tensor(tokenizer.encode(text).ids).unsqueeze(0).to('cuda')

def decoding(ids, tokenizer):
    return tokenizer.decode_batch(ids)

def get_answer(input_sent, model, tokenizer, e_s, unk):
    input_ids = encoding(input_sent, tokenizer)

    sample_outputs = model.generate(
        input_ids,
        num_return_sequences=5,
        do_sample=True, 
        max_length=128, 
        top_k=50, 
        top_p=0.95, 
        eos_token_id=e_s,
        early_stopping=True,
        bad_words_ids=[[unk]]
    )

    decoded_result = decoding(sample_outputs.tolist(), tokenizer)
    # for result in decoded_result[0]:
    #     print(result)
    return decoded_result[1]


def main():
    st.title('KoGPT2 Chatbot \n무엇이든 얘기해주세요')
    # with open("config.yaml") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    tokenizer = SentencePieceBPETokenizer("./kogpt2/vocab.json", "./kogpt2/merges.txt")
    tokenizer.add_special_tokens(['<s>', '</s>'])
    tokenizer.no_padding()
    tokenizer.no_truncation()
    e_s = tokenizer.token_to_id('</s>')
    unk = tokenizer.token_to_id('<unk>')


    torch.manual_seed(42)
    model_dir = '/opt/ml/Kogpt2_chatbot/chitchat_model.bin'
    config = GPT2Config(vocab_size=50000)
    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load(model_dir, map_location='cuda'), strict=False)
    model.to('cuda')

    with st.form(key='입력 form'):
        input_sent = st.text_input('무엇이든 얘기해주세요 귀 기울여드릴게요')
        st.form_submit_button("enter")

    answer = get_answer(input_sent, model, tokenizer, e_s, unk).replace(input_sent, '')
    st.write(answer)

if __name__ == "__main__":
    main()




