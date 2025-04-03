# 🎶 MobileBERT를 활용한 스포티파이 리뷰 분석  🎶


###  긍정 혹은 부정을 예측하는 인공지능 모델을 개발
![image](https://github.com/user-attachments/assets/a4d35ca1-1626-4ad4-a7f4-500600a710b4)



---

## 🎵1. 서론

### 1.1 음악 서비스 시장

음악 스트리밍 서비스는 최근 몇 년 간 급성장하고 있으며, 다양한 음악을 손쉽게 접근할 수 있는 플랫폼들이 늘어나고 있습니다. Spotify와 같은 서비스는 전 세계적으로 수억 명의 사용자들에게 인기를 끌고 있습니다. 이러한 서비스들은 단순히 음악을 스트리밍하는 것에 그치지 않고, 사용자 리뷰, 평가, 추천 시스템 등 다양한 기능을 제공합니다.


![image](https://github.com/user-attachments/assets/9ec2db8e-9f3f-4a8c-83fe-b831dbbe7ee4)


### 1.2 문제 정의

음악 서비스의 확산과 함께 사용자 리뷰는 서비스 품질 향상에 중요한 역할을 합니다.
긍정적인 리뷰는 서비스의 강점과 인기를 보여주며, 부정적인 리뷰는 개선이 필요한 부분을 지적하는 귀중한 피드백이 됩니다.
하지만 방대하게 쌓이는 리뷰를 일일이 분석하는 것은 비효율적이며, 서비스 운영자가 즉각적인 조치를 취하기 어렵습니다.
이 프로젝트는  Spotify 사용자 리뷰를 분석 및 수집하고, 자동으로 리뷰의 감정을 분류하고 긍정적인 리뷰와 
부정적인 리뷰를 예측하는 MobileBERT 기반 감성 분석 모델을 개발하는 것을 목표로 합니다.
이를 통해 서비스의 강점을 더욱 강화하고, 개선이 필요한 부분을 효과적으로 파악하여 서비스 품질을 향상시키고자 합니다.






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

- 이 그래프를 통해 1점 리뷰가 압도적으로 많아 부정적인 피드백이 많이 존재함을 확인할 수 있으며,
전반적으로 낮은 점수가 높은 점수보다 많아 서비스에 대한 사용자 불만이 많을 가능성이 있음을 시사합니다.

  
## 데이터 전처리
- **중복 리뷰 제거**: 동일한 내용의 리뷰 제거
- **텍스트 정제**:특수문자, HTML 태그, 중복 공백 제거
- **토큰화 및 불용어(stopwords) 제거**: 의미 없는 단어 제거
- **라벨링** : 4, 5인 리뷰에는 긍정(1) 1, 2인 리뷰에는 부정(0)을 부여, 평점3은 제외 

## 분석 데이터 구성 
![Figure_12222](https://github.com/user-attachments/assets/b937f01b-e477-4303-823b-15ebaafb0803)


- 여전히 긍정적인 리뷰 보다 부정적인 리뷰가 압도적으로 많은것을 다시 확인 할 수 있었습니다.
- 분석의 정확도를 높이기 위해 긍정적인리뷰와 부정적인 리뷰의 데이터를 각각 15000개씩추출, 총 30,000개의 학습데이터로 재구성 하였습니다.




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

[[Spotify Dataset - Kaggle](https://www.kaggle.com/datasets/alexandrakim2201/spotify-dataset)](https://www.kaggle.com/datasets/ashishkumarak/spotify-reviews-playstore-daily-update?resource=download)

---
