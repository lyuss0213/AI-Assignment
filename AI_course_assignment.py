# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로딩 및 전처리
df = pd.read_csv('C:/Users/user/Downloads/topuniversities.csv')

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

# 1. 교차 검증(K-Fold Cross Validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, Y, cv=kf, scoring='r2')

# 교차 검증 결과 출력
print(f"교차 검증 R² 점수: {cv_scores}")
print(f"평균 교차 검증 R² 점수: {cv_scores.mean():.4f}")

# 2. 부트스트래핑(Bootstrapping)
boot_scores = []

# 부트스트랩 샘플 생성 및 학습
n_iterations = 1000  # 부트스트랩 반복 횟수
for i in range(n_iterations):
    # 데이터 재샘플링
    X_resample, Y_resample = resample(X, Y, random_state=i)
    
    # 모델 학습 및 평가
    model.fit(X_resample, Y_resample)
    y_pred = model.predict(X)
    score = r2_score(Y, y_pred)
    boot_scores.append(score)

# 부트스트래핑 결과 출력
print(f"부트스트래핑 평균 R² 점수: {np.mean(boot_scores):.4f}")
print(f"부트스트래핑 점수 표준편차: {np.std(boot_scores):.4f}")
