import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import  TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

GPU = torch.cuda.is_available()

device = torch.device("cuda" if GPU else"cpu")
print("Using device: ", device)

#1. 학습 시 경고 메세지 제거
logging.set_verbosity_error()

#2. 데이터 확인
path = "spotify_train.csv"
df = pd.read_csv(path, encoding="utf-8")
data_X = list(df["content"].values)
labels =df['Sentiment'].values


print("### 데이터 샘플 ###")
print("리뷰 문장 : ", data_X[:5])
print("긍정/부정 : ", labels[:5])

# 3. 텍스트 를 토큰 으로 나눔(토큰화)
tokenizers = MobileBertTokenizer.from_pretrained('mobilebert_uncased', do_lower_case=True)
#길이 256까지
inputs = tokenizers(data_X,truncation=True, max_length=256, add_special_tokens=True, padding="max_length")#[PAD]
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

num_to_print = 3
print("\n###토큰화 결과 샘플###")
for j in range(num_to_print):
    print(f"\n{j+1}번째 데이터")
    print("데이터 : ", data_X[j])
    print("토큰 : ",input_ids[j])
    print("어텐션 마스크 :", attention_mask[j] )
# 4. 학습용 및 검증용 데이터셋 분리 (scikit learn에 있는 train_test_split 함수 사용, random_state는 반드시 일치시킬 것)
train, validation, train_y, validation_y = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_mask, validation_mask, _, _ = train_test_split(attention_mask, labels, test_size=0.2 , random_state=2025)

# 5. MobileBERT에 영화 리뷰 데이터를 Finetuning하기 위한 데이터 설정
#batch size는 한 번에 학습하는 데이터의 양
batch_size = 8

# 학습용 데이터로더 구현 (torch tensor)
train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks = torch.tensor(train_mask)
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# 검증용 데이터로더 구현
validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y)
validation_masks = torch.tensor(validation_mask)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

#6. 모델 설정
model = MobileBertForSequenceClassification.from_pretrained('mobilebert_uncased', num_labels=2)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)

# 7. 학습(loss), 검증
epoch_results = []

for e in range(epochs):
    # 학습 루프
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {e+1}", leave=True)
    for batch in progress_bar:
        batch_ids, batch_mask, batch_labels = batch

        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        model.zero_grad()

        # 앞먹임 : forward pass
        output = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = output.loss
        total_train_loss += loss.item()

        #역전파 : backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss':loss.item()})

    # 학습 데이터셋에 대한 평균 손실값 계산
    avg_train_loss = total_train_loss / len(train_dataloader)

    # 학습 데이터셋에 대한 정확도(accuracy)계산
    model.eval()
    train_pred = []
    train_true = []

    for batch in tqdm(train_dataloader, desc=f"Evaluation Train Epoch {e+1}", leave=True):
        batch_ids, batch_mask, batch_labels = batch

        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)
        logits = output.logits
        pred = torch.argmax(logits, dim=1)
        train_pred.extend(pred.cpu().numpy())
        train_true.extend(batch_labels.cpu().numpy())

    train_accuracy = np.sum(np.array(train_pred) == np.array(train_true))/len(train_pred)

    #검증 데이터셋에 대한 정확도(accuracy)계산
    val_pred = []
    val_true = []

    for batch in tqdm(validation_dataloader, desc=f"Evaluation validation Epoch{e+1}", leave=True):
        batch_ids, batch_mask, batch_labels = batch

        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)
        logits = output.logits
        pred = torch.argmax(logits, dim=1)
        val_pred.extend(pred.cpu().numpy())
        val_true.extend(batch_labels.cpu().numpy())
    val_accuracy = np.sum(np.array(val_pred) == np.array(val_true))/len(val_pred)

    epoch_results.append((avg_train_loss, train_accuracy, val_accuracy))

# 8. 학습 종료후 epoch별 학습 경과 및 검증 정확도 출력
for idx,(loss, train_acc, val_acc) in enumerate(epoch_results, start=1):
    print(f"Epoch {idx}: Train loss: {loss:.4f}, Train Accuracy:{train_acc:.4f},Validation Accuracy:{val_acc:.4f}")


#9.모델 저장
print("\n### 모델저장 ###")
save_path = "mobilebert_custom_model_imdb"
model.save_pretrained(save_path+'.pt')
print("모델 저장 완료")

import matplotlib.pyplot as plt

# 학습 결과 리스트(epoch_results)를 바탕으로 epoch, train_loss, train_acc, val_acc 추출
epochs = range(1, len(epoch_results)+1)
train_loss = [x[0] for x in epoch_results]
train_acc = [x[1] for x in epoch_results]
val_acc = [x[2] for x in epoch_results]

# 1행 2열의 서브플롯 생성
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# -------------------------------
# (1) 학습 손실(train loss) 그래프
# -------------------------------
axes[0].plot(epochs, train_loss, marker='o', color='blue', label='Train Loss')
axes[0].set_title("train\n tag: log_lm/train")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

# -------------------------------
# (2) 검증 정확도(validation accuracy) 그래프
# -------------------------------
axes[1].plot(epochs, val_acc, marker='o', color='red', label='Validation Accuracy')
axes[1].set_title("valid\n tag: log_lm/valid")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(True)

# 레이아웃 정리 & 그래프 표시
plt.tight_layout()
plt.show()