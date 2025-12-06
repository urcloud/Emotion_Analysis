# 텍스트 기반 감정 분류 모델 개발 프로젝트

## 목표
본 텍스트 기반 감정 분류 모델 개발 프로젝트는 한국어와 영어로 작성된 다양한 텍스트 데이터를 입력받아 감정을 자동으로 분석하는 모델을 구축하는 것을 목표로 한다.

## 설명
여러 데이터셋을 통합하여 다국어 학습 데이터셋을 구성하고, 전통적인 BoW(Bag-of-Words) 기반 모델과 최신 Transformer 기반 모델을 각각 구현하여 성능을 비교했습니다. 다국어 환경에서도 안정적으로 감정을 분류할 수 있는 모델을 개발하고, 두 접근 방식의 정확도와 효율성을 분석하는 데 중점을 두었습니다.

## 데이터셋
### 데이터셋 출처
* https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset
* https://www.kaggle.com/datasets/parulpandey/emotion-dataset

### 데이터셋 개요
* 두 데이터셋을 하나의 파일인 emotion_recognitions_merged.csv 파일로 합쳐서 사용
* text, label 컬럼으로 이루어져 있음
* label의 경우 sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)로 이루어져 있음