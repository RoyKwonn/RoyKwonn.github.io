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

$$ 가중합_{3} = w_{31}y_{h1} + w_{32}y{h2} + 1(bias) $$

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
