---
layout: post
title: "퍼셉트론 (Perceptron)"
categories: deeplearning
author:
- Seokhwan Kwon
meta: "Springfield"
---

## Perceptron이란?

필자는 'Perceptron'이라는 단어를 듣자마자
이렇게 생각했다. Per(완전히) + cep(잡다, 이해하다) + tron(The Realtime Operating System Nucleus)

tron이 좀 어려운데... 네이버 사진을 캡쳐해왔다.
![사전_tron](/assets/images/사전_tron.png)

즉, 완전히 이해하기위한 것이다.

### Perceptron의 역할 및 용어

일반적으로 뇌의 연산구조는 `뉴런 -> 시냅스 -> 뉴런`이렇게 구성되어 있다.
그렇다면, 컴퓨터의 연산구조는 `입력값 -> 활성함수 -> 출력값`이렇게 구성할 수 있지 않는가?

우리는 코딩을 통해서 함수에 reference value를 주면, return value를 받는것을 알고 있다. 저 위의 구조와 동일하지 않는가? 어렵지 않다.


$$ y = ax + b = wx + b $$
>위 식은 x의 값(input)에 따라서 y의 값(output)이 달라지는 것이다.
>쉽게 이해가 가지 않는가?
>
>그렇다면 딥러닝에서의 'wx + b'로 나타낼 수 있는데 이것을 판단함수 또는 활성화 함수(Activation function)이라고 합니다. (ex. 시그모이드)
>
>이때, w와 b는 각각 무엇을 의미하는지 아시나요?

w : weight (가중치)
b : bias (편향, 선입견)
y : weighted sum (가중합)

>위 용어는 매우 중요함으로 잘 기억해두시길 바랍니다.



### Perceptron의 위치

뇌 : 뉴런 -> 신경망 -> 지능
컴퓨터 : Perceptron -> 인공신경망 -> 인공지능

### Perceptron의 구조

![Perceptron_Model](https://miro.medium.com/max/645/0*LJBO8UbtzK_SKMog)

일반적으로 Perceptron의 구조는 이렇게 생겼다.

우리는 앞서 XOR을 바로 연산하기 어렵다는 것을 배운 적이 있다.

![Perceptron_logic](http://ecee.colorado.edu/~ecen4831/lectures/xor2.gif)


### Perceptron의 은닉층

그래서 은닉층이라는 것을 활용하게 되었다.
이것은 수학에서 치환의 개념으로 생각하면 쉽다. (ex. 극좌표 변환)

![Multi_Perceptron](/assets/images/Multi_Perceptron.jpg)

이런식으로 Multi Perceptron을 구성하여 해결할 수 있다.
위 식을 보면 상당히 복잡해 보이는데... 사실 하나하나 알고보면 그리 어렵지 않다.

$$ w(1) = \begin{pmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{pmatrix} = \begin{pmatrix}
-2 & 2 \\
-2 & 2
\end{pmatrix} $$
$$ w(2)= \binom{w_{31}}{w_{32}} = \binom{1}{1} $$
$$ b(1)= \binom{b_{1}}{b_{2}} = \binom{3}{-1} $$
$$ b(2) = b_{3} =(-1) $$

각각 이렇게 나타낸다면.
풀이방식은 아래와 같다.

1. n<sub>1</sub>, n<sub>2</sub>를 구한다.
$$ \binom{n_{1}}{n_{2}} = \begin{pmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{pmatrix} \binom{x_{1}}{x_{2}} + \binom{b_{1}}{b_{2}}
= \binom{w_{11}x_{1} + w_{12}x_{2} + b_{1}}{w_{21}x_{1} + w_{22}x_{2} + b_{2}} $$

2. `1.`에서 구한 n<sub>1</sub>, n<sub>2</sub>를 대입하여, y<sub>out</sub>를 구한다.

$$ y_{out}=  \binom{n_{1}}{n_{2}} \binom{w_{31}}{w_{32}} + b_{3} = n_{1}w_{31} + n_{2}w_{32} + b_{3} $$

위 예시를 가지고 한번 풀어보아라.
그러면 일련의 수식이 나오게 되는데 x<sub>1</sub>, x<sub>2</sub>의 각각 값을 대입해보면 XOR의 연산을 할 수 있을 것이다.

> 위 예제가 바로 Perceptron의 은닉층을 활용하여 XOR를 구하는 기법이다.
> 여기서 은닉층은 n<sub>1</sub>, n<sub>2</sub>이다.

그렇다면, w(1), w(2), b(1), b(2)는 어떻게 구해야 하는 것일까?
우리는 이미 주어진 값들을 가지고 풀어보았지만, 실제로는 이 값들을 알아내야 한다.

알아내는 방법은 다음에 배울 "오차 역전파(Back-propagation)"에서 알 수 있다.
