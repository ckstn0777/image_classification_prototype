## 이미지 분류 프토로타입
### 목표
- 전이학습 되어진 모델(ex. mobilenet 등)을 사용해서 이미지 분류 테스트
- bentoml을 사용해서 api 통신 가능하게 만듬
- 이후 전이학습 되어진 모델을 수정해서 사용해보도록

### 실행환경
- Python 3.8
- tensorflow-2.4.1-py3-none-any.whl (mac 용)
- imageio 2.8.0 (버전 주의)


### 실행
- 가장 먼저 python main.py 실행
- 서비스 실행 : bentoml serve ImageClassifier:latest --port 5001
- 단일 이미지 테스트 목적 : bentoml run ImageClassifier:latest predict --input-file ./images/cat.jpeg