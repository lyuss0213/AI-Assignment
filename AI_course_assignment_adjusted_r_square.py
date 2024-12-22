# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로딩 및 전처리
df = pd.read_csv("D:/USER-DATA/Downloads/topuniversities.csv")

# 독립 변수 및 종속 변수 선택
X = df.loc[:, ['Citations per Paper', 'Papers per Faculty',
       'Academic Reputation', 'Faculty Student Ratio', 'Staff with PhD',
       'International Research Center', 'International Students',
       'Outbound Exchange', 'Inbound Exchange', 'International Faculty',
       'Employer Reputation']].values

Y = df.loc[:, 'Overall Score'].values

# 결측값 대체 (평균값으로)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y.reshape(-1, 1)).ravel()

# 모델 선언
model = LinearRegression()

# K-Fold Cross Validation 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

adjusted_r2_scores = []

# 데이터 크기
n = X.shape[0]  # 전체 샘플 수
p = X.shape[1]  # 특성 수

for train_index, test_index in kf.split(X):
    # 학습 및 테스트 데이터 분할
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    # 예측 및 R^2 계산
    y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, y_pred)
    
    # Adjusted R^2 계산
    adjusted_r2 = 1 - ((1 - r2) * (len(Y_test) - 1) / (len(Y_test) - p - 1))
    adjusted_r2_scores.append(adjusted_r2)

# 결과 출력
print(f"K-Fold Cross Validation Adjusted R^2 점수: {adjusted_r2_scores}")
print(f"평균 Adjusted R^2 점수: {np.mean(adjusted_r2_scores):.4f}")

bootstrapped_adjusted_r2_scores = []
n_iterations = 1000  # 부트스트랩 반복 횟수
n = X.shape[0]  # 전체 샘플 수
p = X.shape[1]  # 특성 수

for i in range(n_iterations):
    # 부트스트랩 샘플 생성
    X_resample, Y_resample = resample(X, Y, random_state=i)
    
    # 데이터 분할
    X_train, X_test, Y_train, Y_test = train_test_split(X_resample, Y_resample, test_size=0.3, random_state=i)
    
    # 모델 학습
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    # R^2 계산
    r2 = r2_score(Y_test, y_pred)
    
    # Adjusted R^2 계산
    adjusted_r2 = 1 - ((1 - r2) * (len(Y_test) - 1) / (len(Y_test) - p - 1))
    bootstrapped_adjusted_r2_scores.append(adjusted_r2)

# 결과 출력
print(f"부트스트래핑 평균 Adjusted R^2 점수: {np.mean(bootstrapped_adjusted_r2_scores):.4f}")
print(f"부트스트래핑 Adjusted R^2 점수 표준편차: {np.std(bootstrapped_adjusted_r2_scores):.4f}")
