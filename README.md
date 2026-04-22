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
├─ notebooks/
│  └─ skhynix_stock_analysis_bilstm.ipynb
├─ requirements.txt
└─ README.md
```

---

## 🔍 분석 흐름 (Section 1 ~ 10)

| 단계 | 분석 과정 | 상세 내용 |
| :---: | :--- | :--- |
| **Section 1** | **데이터 수집** | `yfinance` API를 활용하여 SK하이닉스 일봉 데이터(OHLCV) 로드 및 결측치 제거 |
| **Section 2** | **탐색적 데이터 분석 (EDA)** | 기초 통계량 검토 및 데이터 타입 확인 |
| **Section 3** | **기술적 지표 계산** | 이동평균선(20/60/120일), RSI(14일), 볼린저 밴드(20일), MACD 도출 |
| **Section 4** | **종합 대시보드 구성** | `matplotlib`을 활용한 4-Panel 기술적 지표 시각화 |
| **Section 5** | **상관관계 분석** | 변수 간 선형 상관계수 히트맵 도출 |
| **Section 6** | **수익률 분석** | 일간/연간 누적 수익률, 변동성, 샤프 지수, MDD 통계량 산출 (연간 수익률 추정치: +69.88%, 최대 낙폭: -48.20%) |
| **Section 7** | **Bidirectional LSTM 모델링** | 과거·미래 양방향 문맥을 학습하는 딥러닝 아키텍처 설계 (Huber Loss 적용) |
| **Section 8** | **모델 학습 및 예측** | 최근 30일 데이터를 기반으로 향후 30일(D+1 ~ D+30) 주가 궤적 예측 시각화 |
| **Section 9** | **검증 세트 평가** | 실제값과 예측값의 오차율(MAE, RMSE, MAPE) 및 R2 Score 측정 |
| **Section 10** | **분석 요약** | 파이프라인 요약 및 한계점 명시 |

---

💡 회고록 (Retrospective)
데이터 한계 확인: 과거의 가격 및 거래량 데이터(OHLCV)에만 의존한 딥러닝 모델링의 본질적인 한계를 확인했습니다.

모델 성능 평가: 검증 세트의 R2 Score가 음수(-1.1272)로 산출되었으며, 이는 비정형적인 외부 이벤트가 잦은 주식 시장에서 단순 시계열 패턴만으로는 유의미한 미래 예측을 수행하기 어렵다는 점을 시사합니다.

개선 방안 (Future Work): 향후 반도체 시장의 거시 경제 지표(환율, 금리) 및 뉴스 텍스트 데이터를 활용한 감성 분석(Sentiment Analysis) 지수를 파생 변수로 추가하여 모델의 설명력을 개선할 계획입니다.

프로젝트 의의: 본 분석은 주가 예측 모델링의 전체 사이클을 직접 구현하고 평가하는 데 목적이 있으며, 데이터 파이프라인 구축 및 시계열 딥러닝 모델 적용 역량을 증명합니다.
