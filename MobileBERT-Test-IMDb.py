from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
import numpy as np


def main():
    # 모델명
    model = 'mobilebert_custom_model_imdb.pt'

    # Test
    sentence = """ I love this movie. """  # Test 문장
    logits = test_sentence([sentence], model)  # 함수
    print("문장 : ", sentence)

    if np.argmax(logits) == 1:
        print("\nPositive sentence")
    elif np.argmax(logits) == 0:
        print("\nNegatvie sentence")


def convert_sentence(sentence):
    # 토큰화
    tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased', do_lower_case=True)  # mobile bert
    inputs = tokenizer(sentence, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 배치
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_mask)

    return inputs, masks


def test_sentence(sentence, model):
    # 모델 불러오기
    model = MobileBertForSequenceClassification.from_pretrained(model)
    model.eval()

    # 문장 불러오기 및 배치
    inputs, masks = convert_sentence(sentence)
    batch_input_ids = inputs
    batch_input_mask = masks

    # gradient 무시
    with torch.no_grad():
        # Forward
        outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)

    # loss
    logits = outputs[0]

    # numpy화
    logits = logits.numpy()

    return logits


if __name__ == "__main__":
    main()
