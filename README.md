# 📊 SK Hynix Stock Price Analysis & Prediction Portfolio

### SK하이닉스(000660.KS)의 시계열 데이터와 기술적 지표를 분석하고, Bidirectional LSTM 모델을 적용하여 향후 30일간의 주가 흐름을 예측하는 인공지능 및 데이터 엔지니어링 포트폴리오.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 SK하이닉스(000660.KS)의 2020년부터 현재까지의 일봉 데이터를 바탕으로 탐색적 데이터 분석(EDA)과 기술적 지표 분석을 수행하고, Bidirectional LSTM 모델을 활용하여 향후 30일간의 주가 추이를 예측한 데이터 파이프라인 구축 사례입니다.


## 📂 프로젝트 구조 (Project Structure)
```text
skhynix-stock-analysis/
├─ plots/                   # 분석 결과 시각화 이미지 저장 폴더
├─ src/
│  └─ main.py               # 전체 분석 파이프라인 통합 실행 스크립트
├─ .gitignore                      
├─ LICENSE                         
├─ README.md                # 프로젝트 개요 및 가이드 문서
└─ requirements.txt         # 핵심 라이브러리 목록
```

---

## 🔍 분석 흐름 및 시각화 결과 (Section 1 ~ 10)

| 단계 | 분석 과정&emsp;&emsp;&emsp;&emsp; | 상세 내용 및 결과 시각화 |
| :---: | :--- | :--- |
| **Section 1-3** | **데이터 수집 <br>및 EDA** | `yfinance` 데이터 로드 및 기초 통계량 검토 |
| **Section 4-5** | **기술적 지표 <br>및 대시보드** | MA, RSI, Bollinger Bands, MACD 지표 생성 및 시각화<br><br><img src="plots/01_technical_dashboard.png" width="100%"> |
| **Section 6** | **상관관계 분석** | OHLCV 및 기술 지표 간 상관계수 히트맵 도출<br><br><img src="plots/02_correlation_heatmaps.png" width="100%"> |
| **Section 7** | **수익률 분석** | 누적 수익률 및 연간 수익률 분포 분석<br><br><img src="plots/03_volume_return_analysis.png" width="100%"> |
| **Section 8** | **Bi-LSTM 모델 <br>학습** | Huber Loss 기반 학습 곡선 확인<br><br><img src="plots/04_training_history.png" width="100%"> |
| **Section 9** | **미래 주가 <br>예측** | 최근 데이터를 기반으로 향후 30일 주가 궤적 예측<br><br><img src="plots/05_stock_prediction.png" width="100%"> |
| **Section 10** | **모델 평가** | 검증 세트 실제값 vs 예측값 비교 분석<br><br><img src="plots/06_validation_evaluation.png" width="100%"> |

---

##💡 회고록 (Retrospective)

딥러닝을 활용한 주가 예측은 데이터 분석 공부를 시작할 때부터 꼭 도전해보고 싶었던 과제였습니다. 하지만 이번 프로젝트를 통해 데이터의 양보다 중요한 것은 **데이터가 담고 있는 정보의 질**이라는 것을 많이 느꼈습니다.

단순히 과거의 가격과 거래량(OHLCV) 정보만으로 딥러닝 모델을 학습시키는 것은, 마치 시험 범위가 아닌 교과서만 보고 내일의 난이도를 예측하는 것과 같았습니다. 차트 속의 숫자에는 반도체 업황, 거시 경제, 글로벌 공급망 등 시장의 흐름을 결정짓는 핵심 맥락이 빠져 있었습니다. 데이터 엔지니어링 관점에서 더 넓은 범위의 데이터를 수집하고 정제하는 파이프라인 구축의 중요성을 실감했습니다.

검증 세트에서 산출된 음수(-)의 R2 Score를 확인했을 때, 모델이 단순 평균으로 예측하는 것보다도 못한 성능을 보였다는 것은, 단순히 모델의 레이어를 깊게 쌓는다고 해결될 문제가 아니라는 신호였습니다. 주식 시장의 강한 비정형성과 무작위성 앞에서는 정교한 알고리즘 이전에 **도메인 지식에 기반한 피처 엔지니어링**이 선행되어야 함을 배웠습니다.

또한 실패한 예측 결과는 오히려 제가 공부해야 할 방향을 명확히 제시해주었습니다. 비록 예측 정확도는 기대에 미치지 못했지만, 데이터 수집부터 모델 배포 및 시각화까지의 전체 분석 라이프사이클을 설계하고 구현 해냈다는 점에서 큰 의미가 있었습니다. 기술적인 구현 능력을 넘어, 모델의 한계를 객관적으로 인정하고 개선점을 찾아내는 엔지니어의 자세를 가질 수 있었던 값진 경험이었습니다. 

이번 프로젝트를 디딤돌 삼아, 데이터의 복잡성을 이해하고 예측 성능을 개선하기 위해 다음과 같은 방향으로 기술적 깊이를 더해가고자 합니다.
* **멀티모달(Multi-modal) 데이터 구축:** 수치 데이터만으로는 담지 못하는 시장의 심리를 반영하기 위해, 뉴스 헤드라인에 BERT 모델을 적용한 '감성 지수'를 파생 변수로 추가하여 분석의 다각화를 시도하겠습니다.
* **도메인 지식 기반의 피처 엔지니어링:** 반도체 산업의 특수성을 고려하여 필라델피아 반도체 지수(SOX), 환율, 금리 등 거시 경제 지표를 통합하고, 산업 사이클 데이터를 반영하여 모델의 설명력을 개선하겠습니다.
* **아키텍처 현대화 및 파이프라인 자동화:** Transformer 기반의 시계열 모델로 아키텍처를 고도화하고, 데이터 수집부터 예측값 갱신까지 자동으로 수행되는 실시간 데이터 파이프라인 구축에 도전할 계획입니다.

단순히 모델을 돌려보는 것을 넘어, 데이터의 흐름을 설계하고 문제의 본질을 해결하는 엔지니어로 성장하겠습니다.
