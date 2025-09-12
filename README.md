# Mosaic ML 경진대회 (House Price Prediction)

## 👥 Team

| ![박준수](https://avatars.githubusercontent.com/u/156163982?v=4) | ![정무곤](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김수현](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김예인](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오정택](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [박준수](https://github.com/parkjunsu3321) | [정무곤](http://github.com/mugon-jeong) | [김수현](https://github.com/Daisy7942) | [김예인](https://github.com/yeondu-0) | [오정택](https://github.com/Jeong5689) |
| 팀장, Modeling / 코드 통합 | Modeling / Data Preprocessing | EDA / Hyperparameter Tuning | Feature Engineering | Data Preprocessing / EDA |

---

## 0. Overview

### Environment

- Python 3.10  
- Jupyter Notebook  
- LightGBM 4.x  
- Scikit-learn 1.5+  
- Pandas, NumPy, Matplotlib, Seaborn  

### Requirements

```bash
pip install lightgbm scikit-learn pandas numpy matplotlib seaborn
```

---

## 1. Competition Info

### Overview

- 주어진 학습 데이터(train.csv)를 바탕으로 평가 데이터(eval.csv)에 대한 예측을 수행하는 회귀 문제  
- 목표: 주어진 피처(feature)들을 활용해 타깃 변수를 정확히 예측  

### Timeline

- Start Date: 2025-09-01  
- Final Submission: 2025-09-30  

---

## 2. Components

### Directory

```
├── code
│   ├── lgbm_basic.ipynb   # LightGBM 기본 학습 노트북
│   └── train.py           # (추가 예정) 모델 학습 파이썬 스크립트
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
│       └── reference_paper.pdf
└── input
    └── data
        ├── train.csv
        └── eval.csv
```

---

## 3. Data Description

### Dataset Overview

- **train.csv**: 학습용 데이터 (피처 + 타깃)  
- **eval.csv**: 평가용 데이터 (피처만 제공, 예측 제출 필요)  

### EDA

- 결측치 확인 → 결측 없음  
- 변수 분포 확인 → 일부 스케일 차이가 큰 피처 존재  
- 상관계수 분석 → 특정 피처군에서 높은 상관관계 발견  

### Data Processing

- 불필요한 컬럼 제거  
- StandardScaler 적용 (특정 모델에서 활용 가능하도록)  
- train/test split 후 모델 학습  

---

## 4. Modeling

### Model Description

- **LightGBM (LGBMRegressor)**  
  - 빠른 학습 속도와 효율적인 메모리 사용  
  - 대규모 데이터에서도 우수한 성능  
  - 기본 하이퍼파라미터로 시작 후, 추후 Optuna를 통한 튜닝 예정  

### Modeling Process

1. 데이터 로드 및 전처리  
2. train/test split  
3. LGBMRegressor 학습  
4. 예측 및 RMSE 평가  

```python
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_valid)

rmse = mean_squared_error(y_valid, preds, squared=False)
print("Validation RMSE:", rmse)
```

---

## 5. Result

### Leader Board

- Local Validation RMSE: **0.XXXX**  
- Public LB: **Rank X / Score 	12989.6182**  

### Presentation

- [발표 자료 PPT 링크](https://docs.google.com/presentation/d/1PjgTLTGMpGp80vlySwAtpP0xKj7b8I3V/edit?usp=sharing&ouid=116241898945312005453&rtpof=true&sd=true)  

---

## etc

### Meeting Log

- [회의록 (Notion)](https://www.notion.so/4-25240cb3731d800b8ee0f277ad92fc95?source=copy_link)

### Reference

- [LightGBM 공식 문서](https://lightgbm.readthedocs.io/)  
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)  
