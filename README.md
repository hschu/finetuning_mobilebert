# MobileBERT를 활용한 영화리뷰 긍부정 분석

## 1. 들어가며
이 저장소는 도서 **트랜스포머부터 이해하는 생성 인공지능 100제**에 수록된 사전 학습 언어 모델(Pretrained Language Model) 재학습을 다루고 있습니다. 모델은 2020년 구글이 개발한 MobileBERT이며, 소형 모델이기 때문에 노트북이나 PC에서도 재학습이 가능합니다. 이 저장소에서는 세계적인 영화 리뷰 사이트인 IMDb에서 수집된 긍부정 리뷰 데이터를 MobileBERT에 재학습시켜보고 결과를 확인해 보겠습니다. 실행 환경은 통합개발환경(Integrated Development Environment, 이하 IDE)을 사용하며, 여러분들이 한 단계씩 따라할 수 있도록 가이드를 드리고 영상도 첨부합니다. 컴퓨터 프로그래밍을 조금 해보신 분이라면 큰 무리 없이 실행이 가능합니다만, 질문이 있는 경우에는 github 상단의 Issue를 통해 문의하시면 답변을 드리겠습니다.

## 2. 환경 설정
컴퓨터 프로그래밍을 하다보면 환경 설정(configuration)을 하는데 소요되는 시간이 상당합니다. 특히 파이썬과 같은 언어에서는 패키지의 버전에 따라 실행이 되고 안되는 경우가 있습니다. 대부분은 오류 메시지로 인해 작성한 소스코드가 실행되지 않기 때문에, 구글링하여 해법을 찾는 것이 일반적입니다. 오류 메시지는 컴퓨터에 설치되어 있는 각종 프로그램이나 라이브러리에 따라서, 혹은 운영체제에 따라서도 달라질 수 있습니다. 따라서 모든 오류를 처리하고 프로그램을 실행시키기란 쉬운 일은 아닙니다. 이러한 문제로 인해 컨테이너와 같은 기술을 사용할 수 있지만, 그것은 그것대로 사용하는데 있어 진입장벽이 있기 때문에 이 저장소에서는 전통적인 방식인 IDE를 활용해 컴퓨터에서 프로그램을 실행해 보고자 합니다. 이 저장소는 다음의 환경에서 실행되었습니다.
 * 운영체제 : Windows 11
 * 통합개발환경 : PyCharm Community Edition 2024.2.0.1
 * 프로그래밍 언어 : Python 3.12.3

### 2.1 PyCharm IDE 설치
PyCharm Community Edition은 무료로 사용할수 있는 대표적인 파이썬 프로그래밍 도구로 원활한 코딩을 지원해주는 많은 기능을 담고 있습니다. 이 저장소에서는 2024.2.0.1 버전을 사용하며, 최신 버전을 사용해도 무방할 것으로 판단됩니다만, 자동으로 설치되는 파이썬 버전에 따라서 프로그램이 실행되지 않을수도 있습니다. <U>[링크](https://www.jetbrains.com/ko-kr/pycharm/download/other.html)</U>를 따라가시면 나오는 페이지에서 **2024.2.0.1 - Windows (exe)** 를 클릭하고 다운받은 뒤 설치를 진행하면 됩니다. 설치가 완료되면 영상에 따라 진행하면 자동으로 **Python 3.12** 버전이 설치됩니다. 설치하는 시기에 따라서 버전이 변경될 수도 있습니다. 이 경우에는 별도로 Python 설치하고 Interpreter를 수동으로 선택하면 됩니다. PyCharm과 같이 PC에 설치하는 IDE이외에 각종 코딩 플랫폼에서 제공하는 웹기반 IDE도 있습니다만, 대부분 인터넷 연결을 전제로 하고 있다는 점을 고려해야 합니다.

### 2.2 Git을 통한 소스코드 다운로드
PyCharm을 실행하여 새 프로젝트를 만들 때, github 저장소에 있는 소스코드를 그대로 복사하여 프로젝트로 만들 수 있습니다. 이 때 Windows에서 실행할 수 있는 GIT 프로그램을 설치해야하며, 영상을 참고하여 진행하면 됩니다. 

### 2.3 파이썬 패키지 설치
파이썬이라는 프로그래밍 언어는 컴퓨터와 대화하는 형식으로 프로그래밍 할 수 있는 인터프리터(Interpreter) 입니다. 파이썬은 무료로 사용할 수 있다는 점과 조금 더 직관적으로 컴퓨터 프로그래밍을 할 수 있다는 것이 장점입니다. 특히 무료라는 이점 때문에 다양한 개발자들이 유용한 패키지(특정 기능을 구현하기 위한 프로그램의 집합체)를 개발하고 또 다시 무료로 공개하여 선순환 생태계를 만들었습니다. 사용자의 입장에서는 다양한 기능을 구현할 수 있다는 장점이 있으나, 최적화 수준은 패키지 별로 제각각이기 때문에 많은 작업량을 요구하는 경우 문제가 발생할 수 있습니다. 

파이썬은 특히 딥러닝과 궁합이 좋았습니다. 파이썬의 단점은 낮은 계산 효율이었는데, 이것을 거대 테크 기업들이 직접 패키지를 개발하여 효율을 확보했다는 점입니다. 가장 밑바탕이 되는 GPU(Graphical Processing Unit)의 병렬처리 알고리즘부터, 이 병렬처리의 성능적인 손실을 최소화하기 위한 엔지니어링이 추가되어 사용자는 큰 어려움 없이 간소화된 프로그래밍만으로도 복잡한 딥러닝을 구현할 수 있습니다. 이것은 소위 딥러닝 프레임워크로 불리우며 대표적으로는 구글의 Tensorflow와 JAX, 메타의 pytorch가 있습니다. 

huggingface의 언어 모델 라이브러리인 transformers는 다양한 사전 학습 언어 모델을 효율적으로 불러오고 활용할 수 있는 기능들을 지원합니다. 모델 별로 상이한 어휘 사전, 토큰화 알고리즘, 모델 구조, 모델에서 제공하는 기능을 일관된 인터페이스로 제공하여 사용자들은 손쉽게 언어 모델을 다룰 수 있습니다. 

이러한 패키지들은 모두 무료로 공개가 되어있다는 것이 장점이지만, 전술한 환경 설정의 측면에서는 매우 유감스러운 상황이 발생합니다. 패키지의 버전을 모두 최신으로 설치하면 열에 아홉은 오류가 나오게 마련입니다. 적당한 버전을 찾아서 설치하는 것이 중요한데, 여기에는 뾰족한 방법이 없으므로 일단 해보고 수정해가는 것이 일반적입니다. 다음은 MobileBERT를 재학습하기 위해 별도로 설치한 라이브러리 버전입니다. 이 버전은 requirements.txt에 담겨있는 것으로 PyCharm에서 프로젝트를 생성하면 클릭 몇 번으로 바로 설치가 가능합니다.
<center>

| 패키지          | 버전     | 기능                                          |
|--------------|--------|---------------------------------------------|
| pandas       | 2.2.2  | 데이터셋을 다루기 위한 패키지                            |
| numpy        | 1.26.4 | 파이썬에서 수치를 다루기 위한 기본 패키지                     |
| torch        | 2.3.0  | 딥러닝 프레임워크인 pytorch                          |
| transformers | 4.44.2 | 사전 학습 언어 모델의 미세조정을 위해 학습된 모델을 불러오고 사용하는데 활용 |

</center>
