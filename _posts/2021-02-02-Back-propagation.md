---
layout: post
title: "오차 역전파의 계산법 (Back propagation)"
categories: deeplearning
author:
- Seokhwan Kwon
meta: "Springfield"
---

![출력층의_오차_업데이트](/assets/images/출력층의_오차_업데이트.png)

## 1. 출력층의 오차 업데이트

$$ w_{31} (t + 1) = w_{31}t - \frac{d 오차 y_{out}}{dw_{31}} $$

## 2. 오차 공식

`1.`식에서 y<sub>out</sub>에 대한 값은 아래와 같이 구할 수 있다.
$$오차 y_{out} = 오차y_{o1} + 오차y_{o2} $$

>y<sub>o1</sub>, y<sub>o2</sub>는 앞에서 배운 MSE로 나타낸다.

$$ y_{o1} = \frac{1}{2}(y_{t1} - y_{o1})^{2}  $$
$$ y_{o2} = \frac{1}{2}(y_{t2} - y_{o2})^{2}  $$

>계산을 통해 나오는 출력값(output : y<sub>o1</sub>, y<sub>o2</sub>)이 실제값(target : y<sub>t1</sub>, y<sub>t2</sub>)과 같도록 가중치를 조절해야한다.

위 식을 정리하면 아래와 같다.

$$ 오차dy_{out} = \frac{1}{2}(y_{t1} - y_{o1})^{2} + \frac{1}{2}(y_{t2} - y_{o2})^{2} $$


## 3. 체인룰

`2.`에서 구한 y<sub>out</sub>을 w<sub>31</sub>로 편미분하기 위해서는 체인룰을 알아야한다.

체인룰을 알려면 합성함수의 미분을 알아야한다.
합성함수의 미분공식은 아래와 같다.

$$ \left \{ f(g(x)) \right \}' = f'(g(x))g'(x) $$

$$ \frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} $$

이제 기초지식을 쌓았다. 그러면 `1.`에서 구해야하는 dy<sub>out</sub>/dw<sub>31</sub>를 구하는 방법은 아래와 같다.

$$ \frac{d오차y_{out}}{dw_{31}} = \frac{d오차y_{out}}{dy_{o1}} \cdot \frac{dy_{o1}}{d가중합_{3}} \cdot \frac{d가중합_{3}}{dw_{31}} $$

## 4. 체인룰 계산하기
$$ \frac{d오차y_{out}}{dw_{31}} = \frac{d오차y_{out}}{dy_{o1}} \cdot \frac{dy_{o1}}{d가중합_{3}} \cdot \frac{d가중합_{3}}{dw_{31}} $$

>`=` 다음에 나오는 식을 각각  (1), (2), (3)이라고 한다면, 풀이하면 아래와 같다.

#### (1)
$$ \frac{d오차y_{out}}{dy_{o1}} = \left \{오차y_{o1} + 오차y_{o2}  \right \}' $$

>이때, y<sub>o1</sub>에 대한 편미분이기 때문에  y<sub>o2</sub>는 미분하면 0이 되기때문에 무시하고 연산할 수 있다.

$$ = \left \{ \frac{1}{2}(y_{t1} - y_{o1})^{2} \right\}' $$
$$ = (y_{t1} - y_{o1}) \cdot (-1) $$
$$ \therefore \frac{d오차y_{out}}{dy_{o1}} = (y_{o1} - y_{t1}) $$

#### (2)

$$ \frac{dy_{o1}}{d가중합_{3}}$$
위는 `활성화 함수의 미분`이다. 맨위의 flowchart를 확인해보고 이해해보자.

우리는 `활성화 함수`를 `시그노이드`로 예를 들고 풀어보겠다.

$$ \frac{dy_{o1}}{d가중합_{3}} = y_{o1} \cdot (1 - y_{o1})$$

시그노이드의 미분의 증명은 아래와 같다.

$$ \frac{d}{dx}(\frac{1}{1 + e^{-x}}) = \frac{d}{dx}(1 + e^{-x})^{-1} $$
$$ = -(1 + e^{-x})^{-2} \cdot (-e^{-x}) $$
$$ = \frac{e^{-x}}{(1 + e^{-x})^{2}} $$
$$ = \frac{1 + e^{-x} - 1}{(1 + e^{-x})^{2}} $$
$$ = \frac{1}{1 + e^{-x}} - \frac{1}{(1 + e^{-x})^{2}} $$
$$ = \frac{1}{1 + e^{-x}} (1 - \frac{1}{1 + e^{-x}}) $$
$$ (\because \sigma = \frac{1}{1 + e^{-x}}) $$
$$  \therefore \frac{d}{dx}  \sigma (x) = \sigma (x) (1 - \sigma (x) ) $$

#### (3)

$$ \frac{dy_{o1}}{d가중합_{3}} $$

위 식을 구하기 위해서는 가중합<sub>3</sub>을 구해야 한다.

$$ 가중합_{3} = w_{31}y_{h1} + w_{32}y_{h2} + 1(bias) $$

>신경망에서는 bias를 항상 1로 설정한다.
>why? bias는 그래프를 좌우로 움직이는 역할이다.
>시그노이드에서 bias가 1일 때 가장 안정된 예측을 한다.
>따라서 따로 계산할 필요 없이 1로 설정해준다.

flowchart를 보면, n1, n2 노드로 부터 전달된 y<sub>h</sub>값과 w<sub>(2)</sub>값을 통해 만든다.

$$ \frac{dy_{o1}}{d가중합_{3}} = y_{h1} $$

#### (결론)

$$ \frac{d오차y_{out}}{dw_{31}} = \frac{d오차y_{out}}{dy_{o1}} \cdot \frac{dy_{o1}}{d가중합_{3}} \cdot \frac{d가중합_{3}}{dw_{31}} $$

$$ = (y_{o1} - y_{t1}) \cdot y_{o1}(1 - y_{o1}) \cdot y_{h1} $$


## 5. 가중치 업데이트하기

$$ w_{31} (t + 1) = w_{31}t - \frac{d 오차 y_{out}}{dw_{31}} $$

$$ = w_{31}t - (y_{o1} - y_{t1}) \cdot y_{o1}(1 - y_{o1}) \cdot y_{h1} $$

$$ (\because \delta = (y_{o1} - y_{t1}) \cdot y_{o1}(1 - y_{o1})) $$

>위 식에서 delta 부분을 잘 기억해야한다.
>n3(node)의 delta식이라고 한다.

$$ = w_{31}t - \delta \cdot y_{h1} $$


## 6. 은닉층의 오차 업데이트

![은닉층의_오차_업데이트](/assets/images/은닉층의_오차_업데이트.png)

$$ w_{11}(t + 1) = w_{11}t - \frac{d오차 Y_{out}}{dw_{11}} $$

>d 오차 y<sub>h</sub>가 아닌점에 주목!
>
>why? y<sub>h</sub>는 은닉층 안에 위치하므로
>겉으로 들어나지 않기때문에 사용할 수 없다.

$$ \frac{d오차y_{out}}{dw_{11}} = \frac{d오차y_{out}}{dy_{h1}} \cdot \frac{dy_{h1}}{d가중합_{1}} \cdot \frac{d가중합_{1}}{dw_{11}} $$

>`=` 다음에 나오는 식을 각각  (1), (2), (3)이라고 한다면,
>(2), (3)은 앞에 출력층의 오차 업데이트와 동일한 방법으로 진행하면 된다.
>다만, (1)은 풀이방식이 조금 다르다 왜냐하면 위 flowchart에서 보여지는 것처럼 은닉층이기 때문에 중간의 y<sub>h1</sub>을 바로 알 수 없다. 따라서 y<sub>out</sub>의 값에서 유도를 해야하기 때문이다.

#### (2), (3)

$$ \frac{dy_{h1}}{d가중합_{1}} \cdot \frac{d가중합_{1}}{dw_{11}} = y_{h1}(1 - y_{h1}) \cdot x_{1} $$


## 7. 은닉층의 오차계산 방법

#### (1)

$$ \frac{d오차y_{out}}{dy_{h1}} $$

>위 식은 출력층의 오차 업데이트와 다른 방식으로 풀이해야한다.
>
>why? y<sub>h1</sub>에 대해서 미분해야하기 때문이다.

$$ \frac{d오차y_{out}}{dy_{h1}} = \frac{d(오차y_{o1}) + d(오차y_{o2})}{dy_{h1}} = \frac{d오차y_{o1}}{dy_{h1}} + \frac{d오차y_{o2}}{dy_{h1}}$$

>오차 y<sub>o1</sub>가 있는 식을 (a)라하고, 오차 y<sub>o2</sub>가 있는 식을 (b)라고 하자.

(a)

$$ \frac{d(오차y_{o1})}{dy_{h1}} = \frac{d오차y_{o1}}{d가중합_{3}} \cdot \frac{d가중합_{3}}{dy_{h1}}$$

>앞에 있는것을 (a)-1, 뒤에 있는 것을 (a)-2라고 나누어 풀어보자.

(a)-1
$$ \frac{d오차y_{o1}}{d가중합_{3}} =  \frac{d오차y_{o1}}{dy_{o1}} \cdot \frac{dy_{o1}}{d가중합_{3}} $$
$$ = (y_{o1} - y_{t1}) \cdot y_{o1} (1-y_{o1})$$
> 출력층의 오차계산 방법에서 우리는 이 값을 이미 구했다.
> 풀이 방식이 이해가지 않는다면 이전내용을 참고하시오.
>(tip. 앞의 식은 MSE의 미분이다. 뒤의 식은 시그노이드의 미분이다.)

(a)-2

$$ \frac{d가중합_{3}}{dy_{h1}} = w_{31}$$

>y<sub>h1</sub>과 관련된 식은 무엇인가? w<sub>31</sub> 밖에 없다.

(a)의 결론


$$ \frac{d오차y_{o1}}{dy_{h1}} = (y_{o1} - y_{t1}) \cdot y_{o1} (1-y_{o1}) \cdot w_{31} $$

>여기서 w<sub>31</sub>를 제외한 나머지를 보자. 앞에서 본 delta와 동일하지 않는가?
>그래서 다음과 같이 나타낼 수 있다.

$$ \frac{d오차y_{o1}}{dy_{h1}} = (y_{o1} - y_{t1}) \cdot y_{o1} (1-y_{o1}) \cdot w_{31} = \delta y_{o1} \cdot w_{31} $$


(b)

$$ \frac{d오차y_{o2}}{dy_{h1}} = \frac{d오차y_{o2}}{d가중합_{4}} \cdot \frac{d가중합_{4}}{dy_{h1}} $$
$$ = \frac{d오차y_{o2}}{d가중합_{4}} \cdot w_{41} $$
$$ = (y_{o2} - y_{t2}) \cdot y_{o2} (1-y_{o2}) \cdot w_{41}$$
$$ = \delta y_{o2} \cdot w_{41} $$

>(a)의 풀이방식과 유사하다.

#### (1) = (a) + (b)

$$ \frac{d오차y_{out}}{dy_{h1}} = \frac{d오차y_{o1}}{dy_{h1}} + \frac{d오차y_{o2}}{dy_{h1}}$$
$$ = \delta y_{o1} \cdot w_{31} + \delta y_{o2} \cdot w_{41} $$

#### 은닉층의 오차 업데이트 최종 결과

$$ \frac{d오차y_{out}}{dw_{11}} = \frac{d오차y_{out}}{dy_{h1}} \cdot \frac{dy_{h1}}{d가중합_{1}} \cdot \frac{d가중합_{1}}{dw_{11}} $$

$$ = (\delta y_{o1} \cdot w_{31} + \delta y_{o2} \cdot w_{41}) \cdot y_{h1}(1 - y_{h1}) \cdot x_{1}$$

## 8. 델타식

출력층의 오차 업데이트는 아래와 같다.
$$ (y_{o1} - y_{t1}) \cdot y_{o1}(1 - y_{o1}) \cdot y_{h1} $$

은닉층의 오차 업데이트는 아래와 같다.
$$ (\delta y_{o1} \cdot w_{31} + \delta y_{o2} \cdot w_{41}) \cdot y_{h1}(1 - y_{h1}) \cdot x_{1}$$

>위에서 동일한 부분이 보이는가?
>
>하나, out(1 - out)의 형태가 동일하게 존재한다.
>
>둘, 앞에서 출력층의 (y<sub>o1</sub> - y<sub>t1</sub>)는 오차값임을 배웠다. 그런데 은닉층에서도 오차값이 존재하는가? 존재한다! 은닉층은 그 내부를 알 수 없으므로 출력층에서 y<sub>o</sub>값을 가져와 계산을 해야하기때문에 복잡하게 표현된 것 뿐이지 결국 동일하게 오차를 나타내는 부분이 존재한다.
>
>즉, 출력층과 은닉층 모두 '오차 X out(1-out)'의 형태이다. 이러한 형태를 델타식이라고 한다.

델타가 중요한 이유는 한층을 거슬러 올라갈 때마다 같은 형태로 계속 나타나기 때문이다.

아래는 델타식의 표현방법이다. δh를 은닉층의 델타식이라고 할때, 은닉층의 가중치 업데이트는 다음과 같다.

$$ w_{11}(t + 1) = w_{11}t - \delta h \cdot x_{1} $$
