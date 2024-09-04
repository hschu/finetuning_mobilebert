import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import datetime

# 학습 시 경고 메시지 방지
logging.set_verbosity_error()

# 파일 불러오기
path = "imdb_reviews_sample.csv"
df = pd.read_csv(path, encoding="cp949")
data_X = list(df['Text'].values)   # 문장 컬럼
labels = df['Sentiment'].values     # 라벨 컬럼
print("### 데이터 ###")
print("문장")
print(data_X[:5])
print("라벨")
print(labels[:5])

# 데이터 샘플 출력
num_to_print = 3
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased', do_lower_case=True)  # mobile bert
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("\n\n*** 토큰화 ***")
for j in range(num_to_print):
    print(f"\n{j + 1}번째 데이터")
    print("** 토큰 **")
    print(input_ids[j])
    print("** 어텐션 마스크 **")
    print(attention_mask[j])

# 데이터 분리
train, validation, train_y, validation_y = train_test_split(input_ids, labels, test_size=0.2, random_state=2024)
train_masks, validation_masks, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2024)

# 학습 및 검증 데이터 설정
batch_size = 8
train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks = torch.tensor(train_masks)
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y)
validation_masks = torch.tensor(validation_masks)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# 모델 설정
model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=2)  # mobile bert

# 최적화 알고리즘 설정 (AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# epoch 및 scheduler 설정
epoch = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epoch)

for e in range(0, epoch):
    # 학습
    print('\n\nEpoch {:} / {:}'.format(e + 1, epoch))
    print('Training')
    t0 = time.time()    # 시간 초기화
    total_loss = 0      # 총 loss 초기화
    model.train()       # 모델 - 훈련모드
    for step, batch in enumerate(train_dataloader):
        # step 50 단위 정보
        if step % 50 == 0 and not step == 0:
            elapsed_rounded = int(round((time.time() - t0)))
            elapsed = str(datetime.timedelta(seconds=elapsed_rounded))
            print('- Batch {:>5,} of {:>5,}, Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        # batch 데이터 추출
        batch_ids, batch_mask, batch_labels = tuple(t for t in batch)
        # gradient 초기화
        model.zero_grad()
        # Forward
        outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask, labels=batch_labels)
        # loss
        loss = outputs.loss
        total_loss += loss.item()
        if step % 10 == 0 and not step == 0:
            print("step: {:}, loss: {:.2f}".format(step, loss.item()))
        # Backward
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 최적화(AdamW)
        optimizer.step()
        # 학습률 조정
        scheduler.step()

    # 총 loss 계산
    avg_train_loss = total_loss / len(train_dataloader)
    print("평균 학습 오차 : {0:.2f}".format(avg_train_loss))
    print("epoch 학습에 걸린 시간 : {:}".format(str(datetime.timedelta(seconds=(int(round(time.time() - t0)))))))

    # 검증
    print('\nValidation')
    t0 = time.time()    # 시간 초기화
    model.eval()        # 모델 - 검증모드
    eval_loss, eval_accuracy, eval_steps, eval_examples = 0, 0, 0, 0    # 검증값 초기화
    for batch in validation_dataloader:
        # batch 데이터 추출
        batch_ids, batch_mask, batch_labels = tuple(t for t in batch)
        # gradient 무시
        with torch.no_grad():
            # Forward
            outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask)
        # logit
        logits = outputs[0]
        logits = logits.numpy()
        label_ids = batch_labels.numpy()
        # accuracy
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        eval_accuracy_temp = np.sum(pred_flat == labels_flat) / len(labels_flat)
        eval_accuracy += eval_accuracy_temp
        eval_steps += 1
    print("검증 정확도 : {0:.2f}".format(eval_accuracy / eval_steps))
    print("검증에 걸리 시간 : {:}".format(str(datetime.timedelta(seconds=(int(round(time.time() - t0)))))))

# 모델 저장
print('\nSave Model')
save_path = 'mobilebert_custom_model_imdb'
model.save_pretrained(save_path + '.pt')
# torch.save(model, 'model.pt')
print("\nfinish")
