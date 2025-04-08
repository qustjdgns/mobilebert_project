# 🎶 MobileBERT를 활용한 스포티파이 리뷰 분석  🎶


###  긍정 혹은 부정을 예측하는 인공지능 모델을 개발
![image](https://github.com/user-attachments/assets/a4d35ca1-1626-4ad4-a7f4-500600a710b4)



---

## 🎵1. 서론

### 1.1 음악 서비스 시장

음악 스트리밍 서비스는 빠르게 성장하고 있으며, Spotify는 대표적인 글로벌 플랫폼으로 자리 잡았습니다. 이러한 플랫폼에서 사용자 리뷰는 서비스 개선을 위한 중요한 피드백입니다.


![image](https://github.com/user-attachments/assets/9ec2db8e-9f3f-4a8c-83fe-b831dbbe7ee4)


### 1.2 문제 정의

- 리뷰는 양이 방대하고 수시로 쌓이기 때문에 수작업 분석은 비효율적입니다.
- 감성 분석 모델을 활용해 리뷰를 **긍정/부정 자동 분류**함으로써 빠르고 정확한 서비스 개선 포인트 도출이 필요합니다.

### 1.3 문제 확인 및 데이터 분석
- 전체 리뷰 개수 및 감성(긍/부정) 흐름 분석
- 특정 기간 부정 리뷰 급증 → 버전별 부정 리뷰 상위 10개 확인
  

 전체기간 전체 리뷰 수 변화 확인

![image](https://github.com/user-attachments/assets/221da211-9c7c-4dd3-9b05-73774f160de2)

 전체기간 긍정리뷰,부정리뷰의 변화를 확인 

![image](https://github.com/user-attachments/assets/05a1ed6d-63a0-4111-9c73-67e69e5d3339)


-특정기간 부정 데이터 위주로 갑자기 급증한 사실을 확인 할 수 있음

부정적인 리뷰가 특정기간에 왜 급증하엿는지 알아보기위해 특정 기간때 사용자들이 가장 리뷰를 많이 쓴 스포티파이 ver을 확인.

![image](https://github.com/user-attachments/assets/f64408ee-2486-4060-b2f0-db2b01533932)



# App 버전별 부정 리뷰 현황 (2023.10 ~ 2024.05)

앱 버전별로 사용자 부정 리뷰가 집중된 문제점을 요약한 보고서입니다.  
특히, `8.8.86.364` 버전은 단일 버전 기준으로 가장 많은 부정 리뷰를 기록했습니다.

---

## 주요 문제 요약

| 문제 항목 | 설명 |
|-----------|------|
| **앱 오류 및 크래시** | 실행 불가, 강제 종료, 기능 작동 오류 |
| **성능 저하** | 앱 느림, 로딩 지연, 배터리 과소모 등 |
| **UI/UX 변경 불만** | 인터페이스 변경으로 인한 혼란, 접근성 저하 |
| **기능 삭제 또는 제한** | 자주 쓰이던 기능 제거, 기기/지역별 제한 발생 |
| **과도한 광고/결제 유도** | 사용자 경험 방해, 과한 인앱 결제 유도 |
| **호환성 문제** | 특정 OS/기기와의 충돌 또는 실행 오류 |

---

## 최다 부정 리뷰 버전

| 앱 버전 | 부정 리뷰 수 (예시 기준) |
|---------|------------------|
| `8.8.86.364` | **2600+** |
| `8.9.34.590` | 약 1370 |
| `8.8.92.700` | 약 1360 |
| `8.8.88.397` | 약 1250 |
| *(이하 생략)* | ... |

> `8.8.86.364` 버전은 전체 부정 리뷰 중 가장 높은 비중을 차지하며, 빠른 대응이 필요한 주요 대상입니다.

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

[[[Spotify Dataset - Kaggle](https://www.kaggle.com/datasets/alexandrakim2201/spotify-dataset)](https://www.kaggle.com/datasets/ashishkumarak/spotify-reviews-playstore-daily-update?resource=download)](https://www.kaggle.com/datasets/ashishkumarak/spotify-reviews-playstore-daily-update?resource=download)

---
