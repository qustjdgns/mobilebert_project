# 🎶  MobileBERT를 활용한 Spotify 리뷰 감성 분석  🎶

사용자 리뷰를 통해 Spotify 앱의 긍정/부정 감정을 분석하는 인공지능 모델을 개발하였습니다. 본 프로젝트에서는 MobileBERT를 활용해 대규모 리뷰 데이터를 분류하고, 주요 문제 버전을 식별하여 개선 포인트를 도출하는 것을 목표로 하였습니다.


![image](https://github.com/user-attachments/assets/a4d35ca1-1626-4ad4-a7f4-500600a710b4)



---

## 🎵1. 서론

### 1.1 음악 스트리밍 시장과 사용자 리뷰의 중요성

음악 스트리밍 시장은 빠르게 성장하고 있으며, Spotify는 그 중심에 있는 대표적인 글로벌 플랫폼입니다. 이러한 서비스에서 사용자 리뷰는 제품 개선을 위한 핵심 피드백 수단이지만, 방대한 리뷰를 수작업으로 분석하는 것은 비효율적입니다.

![image](https://github.com/user-attachments/assets/9ec2db8e-9f3f-4a8c-83fe-b831dbbe7ee4)


### 1.2 문제 정의

- 수많은 리뷰를 효율적으로 분석하기 위한 자동 감성 분류 시스템이 필요합니다.
- 긍정/부정 리뷰를 자동 분류함으로써 서비스 개선에 필요한 인사이트를 빠르게 도출할 수 있습니다.

### 1.3 데이터 흐름 및 문제 탐색
- 전체 리뷰 수 및 감성 흐름 시각화를 통해 시간에 따른 트렌드를 확인했습니다.
- 특정 기간 동안 부정 리뷰가 급증한 시점을 포착하고, 해당 시기의 앱 버전별 주요 문제를 분석했습니다.
  

 전체기간 전체 리뷰 수 변화 확인

![image](https://github.com/user-attachments/assets/221da211-9c7c-4dd3-9b05-73774f160de2)

 전체기간 긍정리뷰,부정리뷰의 변화를 확인 

![image](https://github.com/user-attachments/assets/05a1ed6d-63a0-4111-9c73-67e69e5d3339)


-특정기간 부정 데이터 위주로 갑자기 급증한 사실을 확인 할 수 있음

부정적인 리뷰가 특정기간에 왜 급증하엿는지 알아보기위해 특정 기간때 사용자들이 가장 리뷰를 많이 쓴 스포티파이 ver을 확인.

![image](https://github.com/user-attachments/assets/f64408ee-2486-4060-b2f0-db2b01533932)


## 주요 문제 요약

#### 📌 App 버전별 주요 문제 요약

| 문제 항목 | 설명 |
|-----------|------|
| 앱 오류 및 크래시 | 실행 불가, 강제 종료 등 |
| 성능 저하 | 느림, 과도한 배터리 사용 등 |
| UI/UX 불만 | 인터페이스 변경으로 인한 혼란 |
| 기능 삭제 | 익숙한 기능 제거, 지역/기기 제한 |
| 광고/결제 유도 | 과도한 인앱 광고/과금 |
| 호환성 문제 | 특정 OS나 기기에서의 오류 |

> 특히 `8.8.86.364` 버전은 부정 리뷰가 2,600건 이상으로, 주요 개선이 필요한 버전으로 분석됨.

## 최다 부정 리뷰 버전

| 앱 버전 | 부정 리뷰 수 (예시 기준) |
|---------|------------------|
| `8.8.86.364` | **2600+** |
| `8.9.34.590` | 약 1370 |
| `8.8.92.700` | 약 1360 |
| `8.8.88.397` | 약 1250 |
| *(이하 생략)* | ... |



---

## 🎵 2. 데이터 확인


## 원본 데이터

![image](https://github.com/user-attachments/assets/af29b164-f260-4bc1-a684-9e8ad61a048c)




- content: 사용자 리뷰의 내용입니다. 예를 들어, 앱의 기능에 대한 불만, 추천, 개선사항 등이 포함됩니다.
- score: 사용자가 부여한 평점입니다. 1부터 5까지의 정수 값으로, 앱에 대한 긍정적 또는 부정적 피드백을 나타냅니다.
- reviewCreatedVersion: 리뷰가 작성된 시점의 앱 버전입니다.
- at: 리뷰가 작성된 날짜와 시간입니다.
- appVersion: 리뷰가 작성된 당시 앱의 버전 정보입니다.


## 원본 데이터 분석
![image](https://github.com/user-attachments/assets/51a1f109-2233-46c3-84e2-e2b14a4258fd)

- 부정적인 리뷰가 긍적적인 리뷰보다 압도적으로 많아 불균형이 심함.


  
##  전처리 과정
- **중복 리뷰 제거**: 동일한 내용의 리뷰 제거
- **텍스트 정제**:특수문자, HTML 태그, 중복 공백 제거
- **토큰화 및 불용어(stopwords) 제거**: 의미 없는 단어 제거
- **라벨링** : 4, 5인 리뷰에는 긍정(1) 1, 2인 리뷰에는 부정(0)을 부여, 평점3은 제외


## 분석 데이터 구성 
![Figure_12222](https://github.com/user-attachments/assets/b937f01b-e477-4303-823b-15ebaafb0803)
→ 긍/부정 각각 15,000개씩 균형 있게 샘플링하여 총 30,000개의 학습 데이터 구성





---

## 🎵 3. 결과

##  3-1. MobileBERT 학습 결과
![Figure_1](https://github.com/user-attachments/assets/c4c951cf-bb75-461d-a9d1-bda3214c29bf)


- 학습 손실 (Train Loss)

  초기 손실 값이 매우 높다가 (약 12,500), Epoch 2 이후 급격히 감소하여 거의 0에 수렴합니다.

- 검증 정확도 (Validation Accuracy)

  초기 정확도(0.935)에서 시작해 Epoch 2 이후 점진적으로 증가하여 0.94에 근접합니다.

  모델이 학습하면서 일관되게 정확도가 증가하는것으로 보아 학습이 잘되고 있다고 볼 수 있습니다

  하지만 Epoch 3 이후 증가 폭이 둔화되므로 학습률 조정이나 추가 데이터 검토가 필요할 수도 있습니다.
  
![스크린샷 2025-04-03 194717](https://github.com/user-attachments/assets/d0f7bc05-d6d6-4de4-84f5-55b2eb3f11bf)
- Epoch 3 이후 과적합 가능성이 높아지고 Epoch 4에서 손실 값이 증가하며, 훈련 정확도는 계속 오르지만 검증 정확도는 멈춰 있어서 모델이 훈련 데이터에 과하게 맞춰져 일반화가 부족합니다.


## 3.2 분석 데이터 전체에 적용한 결과
![ddddd](https://github.com/user-attachments/assets/d0bea670-3dff-454a-b2bb-33b5df2674f7)

![Figure_111](https://github.com/user-attachments/assets/765456dc-cb0e-4447-9529-c2e1bbdc9cd0)
- 최종 정확도(accuracy)가 약 88.51% 달성
- 모델이 부정적 리뷰(13774개)와 긍정적 리뷰(12780개)를 대부분 정확하게 예측하며, 감성 분석에서 높은 정확도를 유지하고 있습니다. 또한, 긍정과 부정 감성을 균형 있게 분류하며 전반적으로 신뢰할 수 있는 결과를 제공합니다. 이러한 성능을 바탕으로 다양한 텍스트 데이터에서 감성 분석을 효과적으로 활용할 수 있을 것으로 기대됩니다.
---

## 🎵 4. 느낀점

이번 프로젝트를 통해 MobileBERT 모델의 감성 분석 성능을 깊이 이해할 수 있었습니다. 모델은 88.51%의 정확도를 달성하며 안정적인 성능을 보였지만, Epoch 3 이후 과적합 가능성이 커져 최적화가 필요함을 느꼈습니다.
또한, 데이터 전처리와 모델의 일반화 성능이 얼마나 중요한지 깨닫게 되었고, 혼동 행렬을 분석하며 성능 개선의 필요성을 실감했습니다. 
전반적으로 유의미한 결과를 얻었으며, 프로젝트를 완성하여 뿌듯합니다.


---

## 참고 문헌

[Engadget - Spotify reaches more than half a billion users](https://www.engadget.com/spotify-reaches-more-than-half-a-billion-users-for-the-first-time-142818686.html)

[[[Spotify Dataset - Kaggle](https://www.kaggle.com/datasets/alexandrakim2201/spotify-dataset)](https://www.kaggle.com/datasets/ashishkumarak/spotify-reviews-playstore-daily-update?resource=download)](https://www.kaggle.com/datasets/ashishkumarak/spotify-reviews-playstore-daily-update?resource=download)

---
