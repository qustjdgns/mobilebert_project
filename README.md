# 🎶 음악 서비스 리뷰 감성 분석 🎶
### MobileBERT를 활용한 긍부정 예측 딥러닝 프로젝트

---

## 🎵1. 서론

### 1.1 음악 서비스 시장

음악 스트리밍 서비스는 최근 몇 년 간 급성장하고 있으며, 다양한 음악을 손쉽게 접근할 수 있는 플랫폼들이 늘어나고 있습니다. Spotify와 같은 서비스는 전 세계적으로 수억 명의 사용자들에게 인기를 끌고 있습니다. 이러한 서비스들은 단순히 음악을 스트리밍하는 것에 그치지 않고, 사용자 리뷰, 평가, 추천 시스템 등 다양한 기능을 제공합니다.

음악 서비스가 확대됨에 따라, 사용자 리뷰는 서비스 개선에 중요한 역할을 합니다. 사용자들이 남긴 긍정적인 리뷰는 서비스의 장점과 인기를 드러내며, 부정적인 리뷰는 개선점이나 문제점을 제시하는 중요한 정보가 됩니다.

문제점을 개선하고 사용자의 만족도가 상승하면 자연스럽게 회사의 실적증가,이용자 증가 등을 기대할 수 있습니다.
![image_readbot_2018_274019_15250473433292396](https://github.com/user-attachments/assets/20cb8502-9d08-483c-968b-517170b22f7a)

### 1.2 문제 정의

본 프로젝트는 이러한 사용자 리뷰 데이터를 기반으로, 음악 서비스에 대한 긍정적인 리뷰와 부정적인 리뷰를 분류하는 감성 분석 모델을 개발하는 것입니다. 감성 분석을 통해 서비스의 사용자 피드백을 보다 쉽게 분석하고, 개선할 부분을 찾을 수 있습니다.







---

## 🎵 2. 데이터

### 2.1 데이터 구성

- **전체 리뷰 수**: 51,473개  
- **긍정적 리뷰 (POSITIVE)**: 22,648개 (44%)  
- **부정적 리뷰 (NEGATIVE)**: 28,825개 (56%)

### 2.2 데이터 예시

- **긍정적 리뷰**:  
  "Great music service, the audio is high quality and the app is easy to use. Also very quick and friendly!"  
  "Please ignore previous negative rating. This app is super great. I give it five stars+"  

- **부정적 리뷰**:  
  "The app crashes constantly, and I keep getting logged out for no reason. Extremely frustrating."  
  "Too many ads and limited skips unless you pay. Not worth it."

### 2.3 데이터 전처리
- **중복 리뷰 제거**: 동일한 내용의 리뷰 제거
- **텍스트 정제**:특수문자, HTML 태그, 중복 공백 제거
- **토큰화 및 불용어(stopwords) 제거**: 의미 없는 단어 제거
- **긍정/부정 분류**: 리뷰의 감성(Label)은 두 가지로 분류됩니다. "POSITIVE"와 "NEGATIVE".
- **데이터 처리**: 각 리뷰는 해당하는 감성 라벨에 따라 전처리되어 모델 학습에 사용됩니다.
  
- 전처리 후 POSITIVE의 개수: 21279개
- 전처리 후 NEGATIVE의 개수: 27423개
---

## 🎵 3. 모델 개발

### 3.1 모델 선택: MobileBERT

MobileBERT는 모바일 환경에서 효율적으로 작동하는 경량화된 BERT 모델로, 자연어 처리에 뛰어난 성능을 보입니다. 본 프로젝트에서는 이 모델을 사용하여 리뷰의 감성을 예측합니다.

### 3.2 학습 데이터 준비

- **훈련 데이터**: 긍정적 리뷰와 부정적 리뷰를 균형 있게 학습 데이터로 구성. 총개수의 80%
- **검증 데이터**: 긍정적 리뷰와 부정적 리뷰를 검증. 총개수의 20%
- **학습 방법**: MobileBERT 모델을 사용하여 각 리뷰에 대해 감성(Label)을 예측합니다. 모델은 입력된 텍스트를 분석하여, 긍정 또는 부정으로 분류하는 방식입니다.

### 3.3 학습 결과

모델 학습을 통해, 훈련 데이터에 대한 정확도(Accuracy)와 성능을 측정합니다. 학습 과정에서 훈련 데이터의 손실(loss)과 정확도(accuracy)를 주기적으로 체크하여, 모델이 잘 학습되고 있는지 확인합니다.

---

## 🎵 4. 결과

### 4.1 훈련 곡선

훈련 데이터에 대한 손실(loss)은 꾸준히 감소하고 있으며, 정확도(accuracy)는 상승하는 추세를 보였습니다. 이를 통해 모델이 잘 학습되고 있음을 확인할 수 있습니다.

![변성훈과제자료](https://github.com/user-attachments/assets/2ef0c09b-62ea-4148-aeeb-f5caa97618af)
![변성훈과제자료2](https://github.com/user-attachments/assets/43343811-d2e4-49cb-a4a7-5c406d8cb261)

### 4.2 모델 성능

모델이 학습한 후, 전체 데이터에 대해 성능을 평가한 결과, 긍정/부정 예측 정확도는 약 **94%**로 나타났습니다. 이는 모델이 대부분의 리뷰에 대해 정확한 감성을 예측한다는 것을 의미합니다.

![스크린샷 2025-03-27 140106](https://github.com/user-attachments/assets/02b6c954-e227-462b-9930-22743413e2e6)





---

## 🎵 5. 느낀점

이번 프로젝트를 통해, 음악 서비스에 대한 사용자 리뷰를 분석하여 긍정적인 감성 및 부정적인 감성을 효과적으로 분류할 수 있음을 확인했습니다. MobileBERT 모델은 텍스트 분석에 있어서 높은 성능을 발휘했으며, 향후 더 많은 데이터와 개선된 전처리 기법을 통해 정확도를 더욱 향상시킬 수 있을 것입니다.

### 향후 개선 사항

- **데이터 불균형 처리**: 데이터셋에서 부정적인 리뷰가 상대적으로 많아 정확도에 영향을 미쳤습니다. 이를 해결하기 위해, 데이터 불균형을 처리하는 방법을 도입할 필요가 있습니다.
- **모델 최적화**: MobileBERT 외에도 다양한 모델을 시도하여 최적의 성능을 달성할 수 있는 방법을 모색할 예정입니다.

---

## 참고 문헌

[Engadget - Spotify reaches more than half a billion users](https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html)

[Spotify Dataset - Kaggle](https://www.kaggle.com/datasets/alexandrakim2201/spotify-dataset)

---

