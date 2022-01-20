import torch
from transformers import BertModel, BertTokenizer
from IPython import embed
from bert_serving.client import BertClient



if __name__ == "__main__":
    bert = BertModel.from_pretrained("data/bert_huggingface")
    tokenizer = BertTokenizer.from_pretrained('data/bert_huggingface')


    def tok(text):
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        return tokens_tensor

    bc = BertClient()
    bert.eval()

    question = "how are you?"
    answer = "I am fine, thank you, and you ?"
    # embed()

    tokenized = tokenizer(text=question, text_pair = answer, return_tensors = "pt")

    bert_result = bert(tokenized.input_ids, tokenized.attention_mask, tokenized.token_type_ids, output_hidden_states=True)
    # embed()
    bert_result = bert_result.hidden_states[-2]
    bc_result = torch.tensor(bc.encode([question + " ||| " + answer])[0][:bert_result.shape[1]])

    mean = torch.mean(bert_result, 0)  # to get the same result with BertClient using pooling strategy REDUCE_MEAN
