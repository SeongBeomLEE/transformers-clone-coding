# Transformers Clone Coding
- Model Serving을 할 때 단순히 pytorch 모델을 바로 사용하기 보다는, 서빙을 더욱더 효율적, 효과적으로 하기 위해서 개발한 모델을 TorchScript 또는 TesorRT 등으로 변환하여 사용하는 경우가 많음
- 변환시에 변환이 안되는 함수가 있을 수 있거나, 모델 자체를 다른 프로그래밍 언어로 변화하여 사용하는 경우도 있을 수 있음 
- 이에 모델 변환 시 모델 구조를 정확하게 이해하는 것은 필수라고 생각하여, Hugging Face의 Transformers에 내재된 Text Model을 클론 코딩함
- 추가로 이번 클론 코딩을 통해서 오픈 소스의 코드 구조 및 모델 구축 방법 등을 많이 배우는 것이 목표임

## 파일 구조
- TODO