import pandas as pd
import torch
import numpy as np
from transformers import MobileBertForSequenceClassification,MobileBertTokenizer
from tqdm import tqdm

GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device: ", device)

data_path = "spotify_inference.csv"
df = pd.read_csv(data_path, encoding="utf-8")





data_X = list(df['content'].values)
labels = df['Sentiment'].values



tokenizers = MobileBertTokenizer.from_pretrained('mobilebert_uncased', do_lower_case=True)
inputs = tokenizers(data_X,truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

batch_size = 8


test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = torch.utils.data.RandomSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model = MobileBertForSequenceClassification.from_pretrained("mobilebert_custom_model_imdb.pt")
model.to(device)

model.eval()

test_pred = []
test_true = []

for batch in tqdm(test_dataloader):
    batch_ids, batch_mask, batch_labels = batch

    batch_ids = batch_ids.to(device)
    batch_mask = batch_mask.to(device)
    batch_labels = batch_labels.to(device)

    with torch.no_grad():
        output = model(batch_ids, attention_mask=batch_mask)

    logits = output.logits
    pred = torch.argmax(logits, dim=1)
    test_pred.extend(pred.cpu().numpy())
    test_true.extend(batch_labels.cpu().numpy())

test_accuracy = np.sum(np.array(test_pred) == np.array(test_true))/len(test_pred)
print(test_accuracy)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 예측값과 실제값을 numpy 배열로 변환
test_pred = np.array(test_pred)
test_true = np.array(test_true)

# 혼동 행렬 계산
cm = confusion_matrix(test_true, test_pred)

# 그래프 그리기
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()