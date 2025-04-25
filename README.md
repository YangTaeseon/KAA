# Perceptive

![Perceptive Banner](images/banner.png)

Perceptive는 AI 모델 개발부터 학습 모델 구축, 인터페이스 개발 및 실험까지 아우르는 올인원 AI 프로젝트입니다.  
Python 기반으로 다양한 AI 툴과 라이브러리를 통합하여 실험 환경을 구성합니다.

---

## 프로젝트 목표

- AI 모델 개발 및 커스터마이징
- 학습 모델 구현 및 실험
- 사용자와 상호작용 가능한 인터페이스 구성
- 연구 및 개발의 반복적 실험 기반 제공

---

## 사용 기술

- Python 3.8+
- PyTorch / TensorFlow (선택형)
- NumPy, Pandas, Matplotlib
- Streamlit or Gradio (인터페이스)
- 기타 AI 유관 라이브러리

---

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/your-username/perceptive.git
cd perceptive


## 가상환경 생성 및 패키지 설치

python -m venv venv
source venv/bin/activate   # Windows는 venv\Scripts\activate
pip install -r requirements.txt

python app.py


## MNIST 예제

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모델 구성
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

