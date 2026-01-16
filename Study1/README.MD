# [Study 1. 이태복] Classifier-Free Guidance inside the Attraction Basin May Cause Memorization

- Title: [Classifier-Free Guidance inside the Attraction Basin May Cause Memorization](https://openaccess.thecvf.com/content/CVPR2025/papers/Jain_Classifier-Free_Guidance_Inside_the_Attraction_Basin_May_Cause_Memorization_CVPR_2025_paper.pdf)
- Authors: Jain, A., Kobayashi, Y., Shibuya, T., Takida, Y., Memon, N., Togelius, J., & Mitsufuji, Y.
- Publication: CVPR 2025

---

---

# Motivation

1. Diffusion Model 에서 두 가지 종류의 memorization 문제가 지속적으로 제기되어 왔음
    - Verbatim memorization : Training 이미지를 거의 그대로 재생성
    - Template memorization : Training 이미지의 구도를 그대로 복사하여 새로운 이미지 생성
    - * 이러한 문제는 생성 모델 전반의 오래된 challenge 이기도 했따!
        - 예: GAN 의 mode collapse → 생성 모델이 다양한 데이터의 분포, 모드(mode)를 제대로 생성하지 못하고 몇 개의 패턴으로만 생성 결과가 집중되는 문제, diversity-quality trade-off
2. Diffusion Memorization의 원인
    - 중복되는 이미지와 반복적인 캡션 (such as the duplication of training images and repeated captions)이 원인으로 알려짐
    - 그러나, 이러한 요인을 제거해도 여전히 발생! → 즉, 모든 원인이 밝혀지지 않았음!
3. Inference 단계에서 적용 가능한 전략 필요성 증가
    - 거대한 Diffusion 모델의 재학습은 비효율적인 Solution
    - 따라서 코어 모델의 재학습 없이 inference 단계에서 적용할 수 있는 접근 방식이 요구됨 (methods that can be applied during inference without requiring computationally intensive retraining of the core
    diffusion models.)
4. 기존 기법들의 한계점
    - Text-to-image (T2I) diffusion models
        - text prompt와 image 사이의 strong association이 memorization을 야기함
        - 특히 “tirgger token”
            - 특정 이미지에 매우 독특하게 붙어 있던 단어나 표현 (overly specific text prompts)
            - 동일, 유사한 이미지-텍스트 쌍이 학습에서 여러번 반복 (data duplication)
            - small dataset fine-tuning (LoRA, DreamBooth 등)
        - 기존의 해결법
            - Prompt 또는 text embedding을 perturbation
            - Cross-attention layer를 perturbation
        - 한계점
            - Text (prompt) -conditional 경우에만 적용 가능
            - 실험된 특정 조건에서만 효과가 있고, 일반화가 잘 되지 않음

---

# Preliminaries

GOTO: 

- [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- https://theaisummer.com/classifier-free-guidance/
- [https://www.haoyizhu.site/blog/guidance_in_diffusion/guidance/](https://www.haoyizhu.site/blog/guidance_in_diffusion/guidance/)

![[https://ffighting.net/deep-learning-paper-review/diffusion-model/classifier-free-guidance/](https://ffighting.net/deep-learning-paper-review/diffusion-model/classifier-free-guidance/)](image.png)

[https://ffighting.net/deep-learning-paper-review/diffusion-model/classifier-free-guidance/](https://ffighting.net/deep-learning-paper-review/diffusion-model/classifier-free-guidance/)

1. Classifier Guided Diffusion
    - Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis.", NeurIPS, 2021.

![image.png](image%201.png)

1. Classifier-free diffusion guidance (CFG)
    - Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance." *arXiv,* 2022.

![[https://medium.com/@baicenxiao/understand-classifier-guidance-and-classifier-free-guidance-in-diffusion-model-via-python-e92c0c46ec18](https://medium.com/@baicenxiao/understand-classifier-guidance-and-classifier-free-guidance-in-diffusion-model-via-python-e92c0c46ec18)](image%202.png)

[https://medium.com/@baicenxiao/understand-classifier-guidance-and-classifier-free-guidance-in-diffusion-model-via-python-e92c0c46ec18](https://medium.com/@baicenxiao/understand-classifier-guidance-and-classifier-free-guidance-in-diffusion-model-via-python-e92c0c46ec18)

| **Feature** | **Classifier-Guided** | **Classifier-Free Guidance** |
| --- | --- | --- |
| Need to train another model? | Yes, a classifier needs to be trained using noisy images. | Not really, for example, CLIP can be used directly for text-to-image tasks. |
| Need to retrain the diffusion model? | No, pre-trained diffusion models are usable as is. | Yes, diffusion needs to be retrained using this method. |
| Control over final output | Can control the generated category. The number of classes the classifier can identify is the number of classes you can control in generation. | Any (almost) condition can be controlled. |
- https://medium.com/@baicenxiao/understand-classifier-guidance-and-classifier-free-guidance-in-diffusion-model-via-python-e92c0c46ec18

---

# Observation

![image.png](image%203.png)

![image.png](image%204.png)

CFG는 diffusion trajectory를 training data의 기억 지점으로 끌어당기는 attraction basin을 만들어내며, 초기 단계에서 CFG를 적용할수록 memorization으로 수렴한다.

1. Key Observation: Memorized samples behave differently during sampling
    - 일반적으로 텍스트는 동일하지만 초기 노이즈(seed)가 다르면 diffusion 모델은 서로 다른 이미지를 생성한다.
    - CFG를 아주 초기 단계(t = T)부터 적용하면  → 샘플링 경로가 attraction basin으로 빨려들어가며 memorized output 발생
    - 하지만 CFG를 충분히 나중 단계(t가 작아진 뒤)에 적용하면 → 동일한 prompt라도 memorization이 발생하지 않음
    - CFG가 diffusion trajectory를 특정 basin 안으로 밀어 넣는 시점이 문제의 핵심이다.
    
2. Attraction Basin
    - Funnel(깔때기)처럼 생긴 영역 (Figure 1)
    - 초기 노이즈에 관계없이 CFG가 적용되면 이 공간 안으로 trajectory가 빨려 들어감
    - Reverse diffusion이 진행될수록 funnel은 좁아짐 → 특정 attractor 주변으로 강하게 수렴
    - Attractor:
        - CFG를 적용하면 다양한 초기 노이즈에서도 같은 지점으로 수렴하는 특정 훈련 이미지 주변의 지점
        - 모델이 “기억한” 이미지 혹은 그 근처의 feature cluster.
    - Attraction Basin
        - CFG를 적용하면 attractor로 수렴하는 state-space 상의 영역

![image.png](image%205.png)

1. 논문에서의 정의
- Definition 1 (Denoiser)
    - $X$ : diffusion trajectory 전체의 sample sapce (이미지 공간)
    - $t \in (0,T]$ :  diffusion time step (0, T]
    - $E$ : embedding space
    - denoiser function $\varphi : X \times (0,T] \times E \rightarrow X,  \varphi (x,t,e)$
    
- Definition 2 (Attractor and attraction basin)
    
    ![image.png](image%206.png)
    
    - Attractor 정의
        - $e_p$ :  prompt embedding
        - $x^{a} \in X$ : attractor, 모델이 기억하고 있는, memorized된 이미지
        - $B_{D}(x^a, \epsilon)$ : $x^a$와 지각적으로 유사한 이미지 샘플 집합 (초기 노이즈와 상관없이) (perceptual ball)
        - $D$ : perceptual distance (LPIPS와 같은 지각적, 시각적 유사도 거리)
    - Attraction basin 정의
        - $\delta$  : 확률적 허용치
    
    ![image.png](image%207.png)
    

- Definition 3 (Transition Point)
    - guidance 크기의 변화 패턴으로 basin 내부/외부를 구분할 수 있음
    - Figure 2 실험 예시
        - t = 779 보다 빨리 CFT 적용 → memorized output
        - t = 759 이후에 CFG → non-memorized output
        - 바로 이 구간이 transition point
    
    ![image.png](image%208.png)
    

---

# Mitigation Strategy

- Static Transition Point
    - 모델마다 basin을 빠져나가는 특정 t (예: 500)을 찾고
    - CFG를 그 이후에 적욯하면 memorization을 방지할 수 있다.
- Dynamic Transition Point
    - 각 샘플, 노이즈마다 다른 transition point를 찾자
    - unconditional 과 conditional guidance 의 차이 $\epsilon_{\theta}(x_t, e_p) - \epsilon_{\theta}(x_t, e_0)$가 local minimum에 도달하는 지점
    - 그 시점 이후 CFG 적용
- Opposite Guidance

![image.png](image%209.png)

![image.png](image%2010.png)

- CFG 방향을 반대로 적용, basin에서 더 빠르게 탈출
- 이후 positive CFG로 전환

→ memorization 없음 (CFG를 너무 빠르게 주는 경우 memorization)

→ 품질 유지 

→ condition alignment 유지 (CFG를 너무 천천히 주는 경우 text alignment 실패)

---

# Experimental Results

![image.png](image%2011.png)

![image.png](image%2012.png)

![image.png](image%2013.png)

- Somepalli, Gowthami, et al. "Understanding and mitigating copying in diffusion models.", *NeuralPS, 2023*.
    
    ![image.png](image%2014.png)
    

---

# ~~Conclusion~~

---

# ~~Limitations~~
