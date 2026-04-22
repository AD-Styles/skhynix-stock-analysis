# # 📊 SK하이닉스(SK Hynix) 주가 분석 포트폴리오
# > **Ticker**: 000660.KS | **기간**: 2020-01-01 ~ 현재  
# > **Reference**: 삼성전자 주가 분석 (2일차) 코드 구조 참조  
# > **분석 목표**: EDA → 기술적 지표 → Bidirectional LSTM 기반 30일 주가 예측
# 
# ---
# ## 📌 분석 흐름
# 1. 데이터 수집 (yfinance)
# 2. EDA (기초통계 · 결측치)
# 3. 기술적 지표 (MA · RSI · 볼린저밴드 · MACD)
# 4. 상관관계 분석
# 5. 수익률 분석
# 6. Bidirectional LSTM 예측 모델
# 7. 예측 결과 시각화 및 평가

# ## ⚙️ Section 1. 라이브러리 설치 및 임포트

# !pip install yfinance --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 13
sns.set_style('darkgrid')

print(f'TensorFlow 버전 : {tf.__version__}')
print(f'GPU 사용 가능   : {bool(tf.config.list_physical_devices("GPU"))}')

# ## 📥 Section 2. 데이터 수집
# > **`yf.download()`** 로 SK하이닉스(000660.KS) 일봉 데이터를 수집합니다.  
# > yfinance >= 0.2.x 에서 반환되는 **MultiIndex 컬럼을 자동으로 평탄화**합니다.

TICKER     = '000660.KS'   # SK하이닉스
START_DATE = '2020-01-01'

print(f'[{TICKER}] 데이터 다운로드 중...')
df_raw = yf.download(TICKER, start=START_DATE, progress=False)

# yfinance 버전별 MultiIndex 대응
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.droplevel(1)

df = df_raw.copy()
df.dropna(inplace=True)

print(f'데이터 크기  : {df.shape[0]}행 x {df.shape[1]}열')
print(f'수집 기간    : {df.index[0].date()} ~ {df.index[-1].date()}')
df.head()

# ## 🔍 Section 3. 탐색적 데이터 분석 (EDA)
# | 항목 | 설명 |
# |------|------|
# | `describe()` | 기술 통계 (평균·표준편차·분위수) |
# | `isnull()` | 결측치 확인 |
# | `dtypes` | 데이터 타입 확인 |

print('=' * 45)
print('  기술 통계')
print('=' * 45)
display(df.describe().round(0))

print('\n[결측치 현황]')
print(df.isnull().sum())

print('\n[데이터 타입]')
print(df.dtypes)

# ## 📐 Section 4. 기술적 지표 계산
# | 지표 | 파라미터 | 의미 |
# |------|----------|------|
# | 이동평균선 (MA) | 20 · 60 · 120일 | 추세 확인 |
# | RSI | 14일 | 과매수(>70) / 과매도(<30) 신호 |
# | 볼린저 밴드 | 20일, ±2σ | 변동성 채널 |
# | MACD | 12, 26, 9일 EMA | 모멘텀 강도 |

# ── 이동평균선 ──
df['MA20']  = df['Close'].rolling(window=20).mean()
df['MA60']  = df['Close'].rolling(window=60).mean()
df['MA120'] = df['Close'].rolling(window=120).mean()

# ── RSI (Relative Strength Index) ──
def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_rsi(df['Close'], 14)

# ── 볼린저 밴드 ──
df['BB_Mid']   = df['Close'].rolling(20).mean()
df['BB_Std']   = df['Close'].rolling(20).std()
df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'] * 100

# ── MACD ──
exp12              = df['Close'].ewm(span=12, adjust=False).mean()
exp26              = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']         = exp12 - exp26
df['MACD_Signal']  = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist']    = df['MACD'] - df['MACD_Signal']

# ── 수익률 ──
df['Daily_Return'] = df['Close'].pct_change()
df['Cum_Return']   = (1 + df['Daily_Return']).cumprod() - 1

print('[지표 계산 완료]')
df[['Close','MA20','RSI','BB_Upper','BB_Lower','MACD']].dropna().tail(5)

# ## 📈 Section 5. 종합 대시보드 (4-Panel)

fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=False)
fig.suptitle('SK Hynix (000660.KS) Technical Analysis Dashboard',
             fontsize=16, fontweight='bold', y=1.01)

# Panel 1: 종가 + 이동평균선
ax1 = axes[0]
ax1.plot(df.index, df['Close'],  label='Close',  color='black',  alpha=0.8, linewidth=1)
ax1.plot(df.index, df['MA20'],   label='MA20',   color='blue',   linewidth=1.2)
ax1.plot(df.index, df['MA60'],   label='MA60',   color='orange', linewidth=1.2)
ax1.plot(df.index, df['MA120'],  label='MA120',  color='red',    linewidth=1.2)
ax1.set_title('Close Price with Moving Averages (MA20 / MA60 / MA120)', fontweight='bold')
ax1.set_ylabel('Price (KRW)')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: 볼린저 밴드
ax2 = axes[1]
ax2.plot(df.index, df['Close'],    label='Close',         color='black',  alpha=0.7, linewidth=1)
ax2.plot(df.index, df['BB_Upper'], label='Upper Band',    color='red',    linestyle='--', linewidth=1)
ax2.plot(df.index, df['BB_Mid'],   label='Middle (MA20)', color='blue',   linestyle='-',  linewidth=1)
ax2.plot(df.index, df['BB_Lower'], label='Lower Band',    color='green',  linestyle='--', linewidth=1)
ax2.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.08, color='gray')
ax2.set_title('Bollinger Bands (20-day, +/-2sigma)', fontweight='bold')
ax2.set_ylabel('Price (KRW)')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: RSI
ax3 = axes[2]
ax3.plot(df.index, df['RSI'], label='RSI(14)', color='purple', linewidth=1)
ax3.axhline(70, color='red',   linestyle='--', alpha=0.7, label='Overbought (70)')
ax3.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
ax3.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), alpha=0.2, color='red')
ax3.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), alpha=0.2, color='green')
ax3.set_ylim(0, 100)
ax3.set_title('RSI (Relative Strength Index, 14-day)', fontweight='bold')
ax3.set_ylabel('RSI')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: MACD
ax4 = axes[3]
ax4.plot(df.index, df['MACD'],        label='MACD',        color='blue',   linewidth=1)
ax4.plot(df.index, df['MACD_Signal'], label='Signal Line', color='orange', linewidth=1)
colors_bar = ['green' if v >= 0 else 'red' for v in df['MACD_Hist']]
ax4.bar(df.index, df['MACD_Hist'], label='Histogram', color=colors_bar, alpha=0.5, width=1)
ax4.axhline(0, color='black', linewidth=0.8)
ax4.set_title('MACD (12, 26, 9)', fontweight='bold')
ax4.set_ylabel('MACD')
ax4.set_xlabel('Date')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ## 🌡️ Section 6. 상관관계 히트맵
# > OHLCV 간 선형 상관계수를 시각화.  
# > 1에 가까울수록 강한 양의 상관 / -1에 가까울수록 강한 음의 상관

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# OHLCV 상관관계
corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('OHLCV Correlation Heatmap', fontweight='bold')

# 기술지표 포함 상관관계
ind_cols = ['Close', 'MA20', 'MA60', 'RSI', 'BB_Width', 'MACD', 'Daily_Return']
sns.heatmap(df[ind_cols].dropna().corr(), annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title('Technical Indicator Correlation Heatmap', fontweight='bold')

plt.tight_layout()
plt.show()

# ## 📊 Section 7. 거래량 및 수익률 분석

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('SK Hynix - Volume & Return Analysis', fontsize=14, fontweight='bold')

# 거래량
ax = axes[0, 0]
vol_colors = ['green' if r >= 0 else 'red' for r in df['Daily_Return'].fillna(0)]
ax.bar(df.index, df['Volume'], color=vol_colors, alpha=0.6, width=1)
ax.set_title('Daily Trading Volume')
ax.set_ylabel('Volume')
ax.grid(True, alpha=0.3)

# 누적 수익률
ax = axes[0, 1]
ax.plot(df.index, df['Cum_Return'] * 100, color='navy', linewidth=1.2)
ax.fill_between(df.index, df['Cum_Return'] * 100, alpha=0.15, color='navy')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Cumulative Return (%)')
ax.set_ylabel('Return (%)')
ax.grid(True, alpha=0.3)

# 일간 수익률 분포
ax = axes[1, 0]
daily_pct = df['Daily_Return'].dropna() * 100
ax.hist(daily_pct, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(daily_pct.mean(), color='red',    linestyle='--',
           label=f'Mean: {daily_pct.mean():.2f}%')
ax.axvline(daily_pct.std(),  color='orange', linestyle='--',
           label=f'Std:  {daily_pct.std():.2f}%')
ax.set_title('Daily Return Distribution')
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 연간 수익률 박스플롯
ax = axes[1, 1]
df_ret      = df['Daily_Return'].dropna().copy()
years       = df_ret.index.year
unique_yrs  = sorted(years.unique())
data_by_yr  = [df_ret[years == y].values * 100 for y in unique_yrs]
bp = ax.boxplot(data_by_yr, labels=unique_yrs, patch_artist=True,
                medianprops=dict(color='red', linewidth=2))
box_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_yrs)))
for patch, c in zip(bp['boxes'], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Annual Return Distribution (Box Plot)')
ax.set_xlabel('Year')
ax.set_ylabel('Daily Return (%)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 핵심 통계
close_arr = df['Close'].values
print('=' * 50)
print('  SK하이닉스 수익률 핵심 통계')
print('=' * 50)
ann_ret = (1 + daily_pct.mean() / 100) ** 252 - 1
ann_vol = (daily_pct.std() / 100) * np.sqrt(252)
sharpe  = ann_ret / ann_vol
mdd     = ((df['Close'] / df['Close'].cummax()) - 1).min() * 100
print(f'  연간 수익률 (추정)   : {ann_ret*100:+.2f}%')
print(f'  연간 변동성          : {ann_vol*100:.2f}%')
print(f'  샤프 지수            : {sharpe:.3f}')
print(f'  최대 낙폭 (MDD)      : {mdd:.2f}%')
print(f'  기간 내 최고가       : {df["Close"].max():,.0f} 원')
print(f'  기간 내 최저가       : {df["Close"].min():,.0f} 원')

# ## 🤖 Section 8. Bidirectional LSTM 예측 모델

SEQ_LEN  = 30   # 입력: 최근 30일
PRED_LEN = 30   # 출력: 미래 30일

# 종가 추출 및 정규화
close_values = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(close_values)

# 시퀀스 생성
X, y = [], []
for i in range(len(scaled) - SEQ_LEN - PRED_LEN + 1):
    X.append(scaled[i : i + SEQ_LEN])
    y.append(scaled[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN].flatten())

X = np.array(X)
y = np.array(y)

# 분할 (80% train / 20% val)
split   = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f'전체 샘플 수  : {len(X)}')
print(f'Train shape  : X={X_train.shape}, y={y_train.shape}')
print(f'Val   shape  : X={X_val.shape},   y={y_val.shape}')

# ### 8-2. 모델 아키텍처
# > **삼성전자 분석 코드와 동일한 Bidirectional LSTM 구조**를 사용합니다.  
# > - `Bidirectional LSTM`: 과거·미래 양방향 문맥 학습  
# > - `BatchNormalization`: 학습 안정화, 수렴 가속  
# > - `Dropout`: 과적합 방지  
# > - `Huber Loss`: MSE 대비 이상치에 강건  

tf.random.set_seed(42)
np.random.seed(42)

model = Sequential([
    # Layer 1 – Bidirectional LSTM (128 units)
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQ_LEN, 1)),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 2 – Bidirectional LSTM (64 units)
    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 3 – LSTM (32 units)
    LSTM(32),
    BatchNormalization(),
    Dropout(0.2),

    # Fully Connected
    Dense(64, activation='relu'),
    Dense(PRED_LEN)   # 30일 출력
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='huber',       # 이상치에 강건
    metrics=['mae']
)

model.summary()

# ### 8-3. 모델 학습

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', patience=5,
                      factor=0.3, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# ### 8-4. 학습 곡선 (Loss / MAE)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('Training History', fontsize=13, fontweight='bold')

axes[0].plot(history.history['loss'],     label='Train Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Val Loss',   color='orange')
axes[0].set_title('Huber Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'],     label='Train MAE', color='blue')
axes[1].plot(history.history['val_mae'], label='Val MAE',   color='orange')
axes[1].set_title('MAE')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ### 8-5. 30일 미래 예측 및 시각화

# 최근 30일 → 미래 30일 예측
last_seq    = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
pred_scaled = model.predict(last_seq, verbose=0)
pred_prices = scaler.inverse_transform(
    pred_scaled.reshape(-1, 1)
).flatten()

today_price  = close_values[-1][0]
last_date    = df.index[-1]
future_dates = pd.bdate_range(
    start=last_date + pd.Timedelta(days=1), periods=PRED_LEN
)

# 시각화
fig, ax = plt.subplots(figsize=(16, 6))

recent = df['Close'].iloc[-90:]
ax.plot(recent.index, recent.values,
        label='Actual Close (90d)', color='steelblue', linewidth=1.5)
ax.plot(future_dates, pred_prices,
        label='Predicted (30d)', color='crimson',
        linestyle='--', linewidth=1.5, marker='o', markersize=3)
ax.axhline(y=today_price, color='gray', linestyle=':',
           alpha=0.8, label=f'Today: {today_price:,.0f} KRW')
ax.scatter(future_dates[0],  pred_prices[0],  color='green',  s=120, zorder=5,
           label=f'D+1:  {pred_prices[0]:,.0f} KRW')
ax.scatter(future_dates[-1], pred_prices[-1], color='purple', s=120, zorder=5,
           label=f'D+30: {pred_prices[-1]:,.0f} KRW')
ax.axvspan(future_dates[0], future_dates[-1], alpha=0.05, color='crimson')

ax.set_title('SK Hynix Stock Price Prediction (Bidirectional LSTM)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (KRW)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 수치 출력
def judge(today, predicted, label):
    diff = predicted - today
    pct  = (diff / today) * 100
    direction = '📈 상승' if diff > 0 else '📉 하락'
    print(f'  [{label}] {predicted:>10,.0f}원 | {diff:>+10,.0f}원 ({pct:+.2f}%) {direction}')

print('=' * 60)
print(f'  기준 (오늘 종가) : {today_price:,.0f} 원')
print('=' * 60)
judge(today_price, pred_prices[0],  '내일 (D+1) ')
judge(today_price, pred_prices[4],  '5일 후     ')
judge(today_price, pred_prices[9],  '10일 후    ')
judge(today_price, pred_prices[-1], '30일 후    ')

# ## 📏 Section 9. 검증 세트 모델 평가
# > 예측값과 실제값의 괴리를 수치로 확인합니다.

# 검증 세트 예측 (D+1 첫 번째 스텝 기준)
y_val_pred_scaled = model.predict(X_val, verbose=0)

y_val_true = scaler.inverse_transform(
    y_val[:, 0].reshape(-1, 1)
).flatten()
y_val_pred = scaler.inverse_transform(
    y_val_pred_scaled[:, 0].reshape(-1, 1)
).flatten()

mae  = mean_absolute_error(y_val_true, y_val_pred)
rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
r2   = r2_score(y_val_true, y_val_pred)
mape = np.mean(np.abs((y_val_true - y_val_pred) / y_val_true)) * 100

print('=' * 50)
print('  검증 세트 평가 지표 (D+1 예측 기준)')
print('=' * 50)
print(f'  MAE  (평균 절대 오차)  : {mae:>10,.0f} 원')
print(f'  RMSE (평균 제곱근 오차): {rmse:>10,.0f} 원')
print(f'  MAPE (평균 절대 % 오차): {mape:>10.2f} %')
print(f'  R2   (결정 계수)       : {r2:>10.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Validation Set Evaluation', fontsize=13, fontweight='bold')

axes[0].plot(y_val_true, label='Actual',    color='steelblue', linewidth=1)
axes[0].plot(y_val_pred, label='Predicted', color='crimson',   linewidth=1, alpha=0.8)
axes[0].set_title('Actual vs Predicted (D+1, Validation Set)')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Price (KRW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

mn = min(y_val_true.min(), y_val_pred.min())
mx = max(y_val_true.max(), y_val_pred.max())
axes[1].scatter(y_val_true, y_val_pred, alpha=0.5, s=20, color='steelblue')
axes[1].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect Prediction')
axes[1].set_title(f'Actual vs Predicted Scatter (R2={r2:.4f})')
axes[1].set_xlabel('Actual Price (KRW)')
axes[1].set_ylabel('Predicted Price (KRW)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---
# ## ✅ Section 10. 분석 요약
# 
# | 항목 | 내용 |
# |------|------|
# | 종목 | SK하이닉스 (000660.KS) |
# | 분석 기간 | 2020-01-01 ~ 현재 |
# | 기술지표 | MA20/60/120 · RSI · 볼린저밴드 · MACD |
# | 예측 모델 | Bidirectional LSTM (128→64→32, Huber Loss) |
# | 예측 구간 | 30 영업일 (약 1.5개월) |
# 
# ### 📌 포트폴리오 활용 시 주의사항
# > ⚠️ 본 분석은 **교육 및 포트폴리오 목적**입니다.  
# > 주가 예측 모델은 실제 투자 판단의 근거로 사용할 수 없습니다.  
# > 과거 데이터 기반 LSTM은 외부 이벤트(실적 쇼크·지정학 리스크)를 반영하지 못합니다.

