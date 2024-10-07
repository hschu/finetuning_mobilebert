# MobileBERT를 활용한 영화리뷰 긍부정 예측
[![실행 영상](./thumb.png)](https://youtu.be/XIEK0ULeeB8)
<div align="center">
이미지를 클릭하면 유튜브 영상으로 이동합니다. 
</div>

## 1. 들어가며
이 저장소는 도서 **트랜스포머로 시작하는 언어 모델과 생성형 인공지능**에 수록된 사전 학습 언어 모델(Pretrained Language Model, 이하 PLM) 재학습을 다루고 있습니다. 모델은 2020년 구글이 개발한 MobileBERT이며, 소형 모델이기 때문에 노트북이나 PC에서도 재학습이 가능합니다. 이 저장소에서는 세계적인 영화 리뷰 사이트인 IMDb에서 수집된 긍부정 리뷰 데이터를 MobileBERT에 재학습시켜보고 결과를 확인해 보겠습니다. 

실행 환경은 통합개발환경(Integrated Development Environment, 이하 IDE)을 사용하며, 여러분들이 한 단계씩 따라할 수 있도록 가이드를 드리고 영상도 첨부합니다. 컴퓨터 프로그래밍을 조금 해보신 분이라면 큰 무리 없이 실행이 가능합니다만, 질문이 있는 경우에는 github 상단의 Issue를 통해 문의하시면 답변을 드리겠습니다.

## 2. 환경 설정
컴퓨터 프로그래밍을 하다보면 환경 설정(configuration)을 하는데 소요되는 시간이 상당합니다. 특히 파이썬과 같은 언어에서는 패키지의 버전에 따라 실행이 되고 안되는 경우가 있습니다. 대부분은 오류 메시지로 인해 작성한 소스코드가 실행되지 않기 때문에, 구글링하여 해법을 찾는 것이 일반적입니다. 오류 메시지는 컴퓨터에 설치되어 있는 각종 프로그램이나 라이브러리에 따라서, 혹은 운영체제에 따라서도 달라질 수 있습니다. 따라서 모든 오류를 처리하고 프로그램을 실행시키기란 쉬운 일은 아닙니다. 

이러한 문제로 인해 컨테이너와 같은 기술을 사용할 수 있지만, 그것은 그것대로 사용하는데 있어 진입장벽이 있기 때문에 이 저장소에서는 전통적인 방식인 IDE를 활용해 컴퓨터에서 프로그램을 실행해 보고자 합니다. 이 저장소는 다음의 환경에서 실행되었습니다.
 * 운영체제 : `Windows 11`
 * 통합개발환경 : `PyCharm Community Edition 2024.2.0.1`
 * 프로그래밍 언어 : `Python 3.12.3`

### 2.1 PyCharm IDE 설치
PyCharm Community Edition은 무료로 사용할수 있는 대표적인 파이썬 프로그래밍 도구로 원활한 코딩을 지원해주는 많은 기능을 담고 있습니다. 이 저장소에서는 2024.2.0.1 버전을 사용하며, 최신 버전을 사용해도 무방할 것으로 판단됩니다만, 자동으로 설치되는 파이썬 버전에 따라서 프로그램이 실행되지 않을수도 있습니다. <U>[링크](https://www.jetbrains.com/ko-kr/pycharm/download/other.html)</U>를 따라가시면 나오는 페이지에서 `2024.2.0.1 - Windows (exe)` 를 클릭하고 다운받은 뒤 설치를 진행하면 됩니다. 설치가 완료되면 영상에 따라 진행하면 자동으로 `Python 3.12` 버전이 설치됩니다. 설치하는 시기에 따라서 버전이 변경될 수도 있습니다. 이 경우에는 별도로 파이썬을 설치하고 Interpreter를 수동으로 선택하면 됩니다. 

PyCharm과 같이 PC에 설치하는 IDE이외에 각종 코딩 플랫폼에서 제공하는 웹기반 IDE도 있습니다만, 대부분 인터넷 연결을 전제로 하고 있다는 점을 고려해야 합니다.

### 2.2 Git을 통한 소스코드 다운로드
PyCharm을 실행하여 새 프로젝트를 만들 때, github 저장소에 있는 소스코드를 그대로 복사하여 프로젝트로 만들 수 있습니다. 이 때 Windows에서 실행할 수 있는 GIT 프로그램을 설치해야하며, 영상을 참고하여 진행하면 됩니다. 

### 2.3 파이썬 패키지 설치
파이썬이라는 프로그래밍 언어는 컴퓨터와 대화하는 형식으로 프로그래밍 할 수 있는 인터프리터(Interpreter) 입니다. 파이썬은 무료로 사용할 수 있다는 점과 조금 더 직관적으로 컴퓨터 프로그래밍을 할 수 있다는 것이 장점입니다. 특히 무료라는 이점 때문에 다양한 개발자들이 유용한 패키지(특정 기능을 구현하기 위한 프로그램의 집합체)를 개발하고 또 다시 무료로 공개하여 선순환 생태계를 만들었습니다. 사용자의 입장에서는 다양한 기능을 구현할 수 있다는 장점이 있으나, 최적화 수준은 패키지 별로 제각각이기 때문에 많은 작업량을 요구하는 경우 문제가 발생할 수 있습니다. 

파이썬은 특히 딥러닝과 궁합이 좋았습니다. 파이썬의 단점은 낮은 계산 효율이었는데, 이것을 거대 테크 기업들이 직접 패키지를 개발하여 효율을 확보했다는 점입니다. 가장 밑바탕이 되는 GPU(Graphical Processing Unit)의 병렬처리 알고리즘부터, 이 병렬처리의 성능적인 손실을 최소화하기 위한 엔지니어링이 추가되어 사용자는 큰 어려움 없이 간소화된 프로그래밍만으로도 복잡한 딥러닝을 구현할 수 있습니다. 이것은 소위 딥러닝 프레임워크로 불리우며 대표적으로는 구글의 `Tensorflow`와 `JAX`, 메타의 `pytorch`가 있습니다. 

`huggingface`의 언어 모델 라이브러리인 `transformers`는 다양한 사전 학습 언어 모델을 효율적으로 불러오고 활용할 수 있는 기능들을 지원합니다. 모델 별로 상이한 어휘 사전, 토큰화 알고리즘, 모델 구조, 모델에서 제공하는 기능을 일관된 인터페이스로 제공하여 사용자들은 손쉽게 언어 모델을 다룰 수 있습니다. 

이러한 패키지들은 모두 무료로 공개가 되어있다는 것이 장점이지만, 전술한 환경 설정의 측면에서는 매우 유감스러운 상황이 발생합니다. 패키지의 버전을 모두 최신으로 설치하면 열에 아홉은 오류가 나오게 마련입니다. 적당한 버전을 찾아서 설치하는 것이 중요한데, 여기에는 뾰족한 방법이 없으므로 일단 해보고 수정해가는 것이 일반적입니다. 다음은 MobileBERT를 재학습하기 위해 별도로 설치한 라이브러리 버전입니다. 이 버전은 `requirements.txt`에 담겨있는 것으로 PyCharm에서 프로젝트를 생성하면 클릭 몇 번으로 바로 설치가 가능합니다.

<div align="center">

| 패키지          | 버전     | 기능                                          |
|--------------|--------|---------------------------------------------|
| `pandas`       | 2.2.2  | 데이터셋을 다루기 위한 패키지                            |
| `numpy`        | 1.26.4 | 파이썬에서 수치를 다루기 위한 기본 패키지                     |
| `torch`        | 2.3.0  | 딥러닝 프레임워크인 pytorch                          |
| `transformers` | 4.44.2 | 사전 학습 언어 모델의 미세조정을 위해 학습된 모델을 불러오고 사용하는데 활용 |

</div>

## 3. MobileBERT를 활용한 영화 리뷰의 긍부정 예측
### 3.1 개요
사전 학습 언어 모델(PLM)은 대부분 모델의 규모가 2B(20억 개)보다 작으며, 가중치가 공개되어 있다는 장점이 있습니다. 따라서 특정한 자연어 처리 과업에 특화된 기능을 구현하기에 적합하지만, 모델이 작기 때문에 일정 수준의 학습 데이터가 필요합니다. 학습 데이터의 양에 대한 구체적인 기준은 없습니다만, 최소 수 천건 수준의 데이터가 요구됩니다. 이 예제에서는 인코더 기반 언어 모델 중 경량화 모델인 MobileBERT를 활용하여, 1,000건의 영화리뷰 긍부정 데이터를 학습시켜보고자 합니다. 이후 학습된 모델을 테스트해보겠습니다.

### 3.2 영화 리뷰 데이터
이 예제에서는 글로벌 경진대회 플랫폼인 Kaggle에서 <U>[공개된 데이터](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)</U>를 활용하고자 합니다. 이 데이터는 영화 리뷰 플랫폼인 [IMDb](https://www.imdb.com/)에 등록된 리뷰에서 5만 건을 추출한 데이터로 구성되어 있습니다. 5만 건의 데이터는 긍정 2만5천 건, 부정 2만5천 건으로 분류됩니다. 이 저장소의 파일 `imdb_reviews_sample.csv`에는 긍정 1천 건, 부정 1천 건을 임의로 추출한 데이터로 MobileBERT의 재학습(fine-tuning의 의미)을 진행합니다. 원본 데이터에서는 긍정과 부정을 각각 `positive`와 `negative`로 표현했는데, 저장소에서 제공하는 학습 데이터는 긍정을 1로 부정을 0으로 대체했습니다.

### 3.3 MobileBERT 재학습
MobileBERT는 2020년 구글에서 개발한 경량 PLM으로, 기존에 개발된 BERT 모델을 지식 증류(Knowldege Distillation)의 기법으로 소형화한 모델 입니다. 특히 모델의 크기가 25.3M(2,530만 개)이기 때문에, 노트북 CPU에서도 재학습이 가능합니다. 

MobileBERT는 `huggingface transformers` 패키지에서 사용할 수 있으며, 사전 학습이 완료된 MobileBERT를 자동으로 다운로드 할 수 있습니다. 오프라인에서 작업을 하기 위해서는 가중치를 다운로드 받으면 되는데, 이 경우 개발자가 직접 공개한 `github` 저장소를 참고할 수도 있지만 편의를 위해서 `huggingface`에서 제공하는 [저장소](https://huggingface.co/google/mobilebert-uncased/tree/main)를 이용할 수 있습니다.

일반적으로 재학습을 위한 방법은 간단한 명령 스크립트를 활용하는 방법과 `Trainer`함수를 사용하는 방식이 있으며, 경우에 따라서는 `Trainer` 대신 학습 과정을 소스코드로 구현할 수도 있습니다. 이 저장소에서는 학습 과정을 소스코드로 구현한 방식을 사용하므로 교육적인 목적으로도 활용할 수 있습니다.

MobileBERT를 활용한 영화 리뷰의 긍부정 예측은 영상에 따라 설치된 PyCharm에 환경 설정을 완료한 뒤 실행을 누르시면 학습이 진행됩니다. 학습된 결과는 별도의 모델로 저장이 되고 `MobileBERT-Test-IMDb.py` 파일을 실행하여 작동이 잘 되는지를 확인할 수 있습니다. 다만 검증 정확도에 따라 입력한 문장이 올바르게 예측되지 않을 수 있습니다.

## 4. 마치며

ChatGPT가 무료로 공개되고 맞춤형 GPTs를 코딩 없이 만들 수 있는 시대에 사전 학습 언어 모델(PLM)을 활용한다는 것은 일견 시대에 뒤떨어져 보이기도 합니다. 또한 메타의 `LLaMA`, 구글의 `Gemma`, 마이크로소프트의 `phi` 등 다양한 소규모 거대 언어 모델(smaller Large Language Model, sLLM)이 오픈소스로 등장하는 시점에서 PLM이 설 자리는 크지 않아 보입니다.

그러나 ChatGPT와 같은 LLM은 플랫폼으로 진화하고 있다는 점에서 온라인 연결이 필수적입니다. 또한 sLLM은 범용성을 확보하기 위해 모델의 크기가 최소 2B(20억 개) 수준인 점을 감안해 보면, PLM은 여전히 매력적인 선택지가 될 수 있습니다. 

PLM은 상대적으로 작은 크기의 모델로도 특정 과업을 수행할 수 있는 능력이 있기 때문에, 온디바이스 인공지능을 개발하거나 GPU가 없는 PC에서 실습할 때 최적의 선택지가 될 수 있습니다. 다만 PLM의 단점은 일정한 규모의 학습 데이터를 구축해야한다는 점입니다. 양에 대한 절대적인 기준은 없습니다만, 제 개인적인 경험이나 논문에 명시된 기준으로 가늠해보면 최소 1,000건 이상의 학습 데이터가 필요합니다. 또한 질적인 측면이 매우 중요한데 학습 데이터가 전체 데이터의 분포를 어느 정도 반영해야 한다는 것입니다. 전체 데이터의 분포를 확인하는 방법은 데이터 자체에 대한 이해가 필요하기 때문에, EDA(Exploratory Data Analysis)와 도메인 지식이 매우 중요하다고 볼 수 있습니다. 제공한 소스코드는 문장 분류 과업 전반에 대응 될 수 있기 때문에, 특정한 학습 데이터를 확보한 경우 데이터를 변경하여 손쉽게 재학습이 가능하니 많이 활용하시기 바랍니다.
