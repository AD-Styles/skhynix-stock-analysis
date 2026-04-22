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

| 단계 | 분석 과정 | 상세 내용 및 결과 시각화 |
| :---: | :--- | :--- |
| **Section 1-3** | **데이터 수집 및 EDA** | `yfinance` 데이터 로드 및 기초 통계량 검토 |
| **Section 4-5** | **기술적 지표 및 대시보드** | MA, RSI, Bollinger Bands, MACD 지표 생성 및 시각화<br><br><img src="plots/01_technical_dashboard.png" width="100%"> |
| **Section 6** | **상관관계 분석** | OHLCV 및 기술 지표 간 상관계수 히트맵 도출<br><br><img src="plots/02_correlation_heatmaps.png" width="100%"> |
| **Section 7** | **수익률 분석** | 누적 수익률 및 연간 수익률 분포 분석<br><br><img src="plots/03_volume_return_analysis.png" width="100%"> |
| **Section 8** | **Bi-LSTM 모델 학습** | Huber Loss 기반 학습 곡선 확인<br><br><img src="plots/04_training_history.png" width="100%"> |
| **Section 9** | **미래 주가 예측** | 최근 데이터를 기반으로 향후 30일 주가 궤적 예측<br><br><img src="plots/05_stock_prediction.png" width="100%"> |
| **Section 10** | **모델 평가** | 검증 세트 실제값 vs 예측값 비교 분석<br><br><img src="plots/06_validation_evaluation.png" width="100%"> |

---

💡 회고록 (Retrospective)
데이터 한계 확인: 과거의 가격 및 거래량 데이터(OHLCV)에만 의존한 딥러닝 모델링의 본질적인 한계를 확인했습니다.

모델 성능 평가: 검증 세트의 R2 Score가 음수(-1.1272)로 산출되었으며, 이는 비정형적인 외부 이벤트가 잦은 주식 시장에서 단순 시계열 패턴만으로는 유의미한 미래 예측을 수행하기 어렵다는 점을 시사합니다.

개선 방안 (Future Work): 향후 반도체 시장의 거시 경제 지표(환율, 금리) 및 뉴스 텍스트 데이터를 활용한 감성 분석(Sentiment Analysis) 지수를 파생 변수로 추가하여 모델의 설명력을 개선할 계획입니다.

프로젝트 의의: 본 분석은 주가 예측 모델링의 전체 사이클을 직접 구현하고 평가하는 데 목적이 있으며, 데이터 파이프라인 구축 및 시계열 딥러닝 모델 적용 역량을 증명합니다.
