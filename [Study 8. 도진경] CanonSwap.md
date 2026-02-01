# [Study 8.도진경] CanonSwap: High-Fidelity and Consistent Video Face Swapping via Canonical Space Modulation

+ [Title] CanonSwap: High-Fidelity and Consistent Video Face Swapping via Canonical
Space Modulation
+ [Publication] 2025.07 (ICCV 2025)
+ [Reference] [paper](https://arxiv.org/abs/2507.02691) | [github](https://github.com/Pixel-Talk/CanonSwap) | [project page](https://luoxyhappy.github.io/CanonSwap/)

<br>
<p align='center'>
    <img width="381" height="297" alt="Image" src="https://github.com/user-attachments/assets/41744256-5648-4a2f-8f63-0b147fcf98ec" />
</p>

## 1. 개요
+ 얼굴 이미지를 canonical space에 projection, 다음 두 정보를 분리
    + appearance
    + motion informantion
+ identity와 motion 속성을 분리하는 Canonical Space
	+ 얼굴 이미지 -> 공간 mapping -> face swap -> 원본 비디오 공간
+ PIM (Partial Identity Modulation) 모듈
	+ partial adaptive weight modification 사용, 원치 않는 영역을 보존
	+ sptial mask 사용, target의 identity 특징을 source 영역에 integration
+ Video Face Swap 평가용 벤치마크 제안
	+ 시선 처리 일치 여부
	+ 시간 흐름에 따른 consistency

<br>

## 3. Method
> 
### 3.1. Canonical Swap Space
> · Canonical Space를 사용한 motion / appearance 분리 
<br>
> · Original Space로의 Swapping
<br>
<p align='center'>
    <img width="769" height="407" alt="Image" src="https://github.com/user-attachments/assets/77a9399a-4fa2-4d6f-94a7-b4e1e5b56e48" />
</p>

+ Motion Encoder를 사용한 Motion extraction, M_(o→c)와 M_(c→o) transform 계산
 <br>: vid2vid 에서 swaping space에 대한 아이디어 착안

+ canonical space에서 swap된 결과물에 Refinement 모듈 적용
 <br>: [3D U-NET](https://arxiv.org/abs/1505.04597)
 <br>: artifact 제거 목적


### 3.2. Partial Identity Modulation
> · Canonical Space에서의 face swapping
<br>
> · 정확하고 효율적인 identity 전달
<br>
<p align='center'>
    <img width="349" height="360" alt="Image" src="https://github.com/user-attachments/assets/a85bcd9b-d801-4f83-a914-2d6618aa0a7b" />
</p>

+ GAN-based 방법에서는 **AdaIN**을 사용한 appearance swap 진행
 <br>: **전체 특징맵**에 적용됨, 불안정안 학습의 원인

+ 얼굴 영역에만 identity modulation을 적용
    + Appendix D에서 빠른 수렴 / 학습 시 adverarial 현상 감소 효과 확인

+ PIM 모듈은 여러 개의 PIM Block으로 구성됨.
    + 각 Block은 두 개의 병렬 layer의 output ( $F_{\mathrm{id}}$,$F_{\mathrm{normal}}$ )을 mask predictor가 생성한 공간 마스크 A를 통해 fusion

$$\left( 
F_{\mathrm{out}} = A \odot F_{\mathrm{id}} + (1 - A) \odot F_{\mathrm{normal}}
\right)$$

1) Standard Convolution Branch $F_{\mathrm{normal}}$
    + 입력 특징맵 $F_{\mathrm{in}}$ 을 컨볼루션 가중치 W 로 처리
2) Modulated Convolution Branch $F_{\mathrm{id}}$
    + Style GAN의 modulation/demodulation 방법 적용
    + source 이미지의 identity 특성을 conv Weight에 적용 
    $s_{id}\cdot W$
    <br>: 즉, source identity 특화 필터
    
    + 가중치에 l2-norm을 적용하여 안정화 ${\sqrt{\sum (s_{\text{id}} \cdot W)^2 + c}}$
    <br>: 분산 유지
    <br>: modulation에서 생길 수 있는 지나치게 큰 변화 방지


$$\left(F_{\text{id}} = \operatorname{Conv}\left( F'_{\text{in}}, \frac{s_{\text{id}} \cdot W}{\sqrt{\sum (s_{\text{id}} \cdot W)^2 + c}} \right)\right)$$


3) mask predictor로 생성된 spatial mask A
    + modulation이 얼굴(눈,코,입) 영역에만 적용되도록 함.

$$\left(\(A = \phi(X).\)\right)$$

얼굴에 대한 세밀한 제어로 자연스러운 외형을 보존, 복잡한 배경이나 포즈 변화 발생 시 artifact를 최소화할 수 있다.


### 3.3 Training And Loss Function
+ warping framework [46](https://arxiv.org/pdf/2011.15126)
    + PIM 모듈과 refinement를 end-to-end로 학습.
+ **$\mathcal{L}_{id}$ Identity Loss**
    + 정확한 identity 전달을 위한 Space 별 identity loss 적용, swapped face와 original face 간 identity 유사도 측정
    + $I^c_{s \to t}$ : Canonical Space 결과 
    <br>: canonical swapped 특징을 decoding한 결과
    + $I^o_{s \to t}$ : Original Space로 warping된 결과 
    <br>: canonical swapped 특징을 original space로 warping 후 decoding
    + $E_{id}$ : pre-trained face recognition model [ArcFace](https://arxiv.org/pdf/1801.07698)

$$\left(
\mathcal{L}_{id} = -\left[\text{Sim}\left(E_{id}(I_s), E_{id}(I^c_{s \to t})\right)+ \text{Sim}\left(E_{id}(I_s), E_{id}(I^o_{s \to t})\right)\right]
\right)$$

+ **$\mathcal{L}_{p}$ Perceptual Loss**
    + structural consistency를 위한 swapped face와 target face 간 feature-level similarity 측정
    + canonical space와 original image space 각각에서 LPIPS를 계산
    + 생성된 얼굴과 타깃 얼굴이 동일하게 보이도록 적용, 시각적 자연스러움 확보
        + canonical space: 얼굴의 기저 geometry, identity consistency
        + original space: 최종 합성된 렌더링 퀄리티, texture alignment
    + canonical space에서 swap된 얼굴이 target의 얼굴 구조/지각적 특징을 잘 보존하는가
        + $I^{C}_{s \rightarrow t}$ : canonical space 에서 source→target swap된 이미지
        + $I^{C}_{t}$ : canonical space에서의 target face
        <br>: canonical feature of target를 decoding해서 얻을 수 있음.
    + original space에서의 swap 결과가 target 얼굴의 특징을 제대로 유지하는가
        + $I^{O}_{s \rightarrow t}$ : original image space에서 source→target 얼굴이 삽입된 결과 영상
        + $I^{O}_{t}$ : original space에서의 target face
    
$$\left(
\mathcal{L}_{p} = \mathcal{L}_{LPIPS}(I^{C}_{s \rightarrow t}, I^{C}_{t}) + \mathcal{L}_{LPIPS}(I^{O}_{s \rightarrow t}, I^{O}_{t})
\right)$$

+ **$\mathcal{L}_{mo}$ Motion Loss**
    + 포즈 및 표정 정확도 보존
    + motion extractor에서 추출한 $E$(facial expression)와 $P$(head pose) 사용
    + canonical space에서 swap 결과의 expression, pose 제거 $P^c$,$E^c$
    + original space에서 swap 결과의 pose,expression이 실제 target 영상과 유사하도록 $P^o$,$E^o$

$$\left(
\begin{equation}
L_{mo} = \left\| P^c_{s \rightarrow t} \right\|_1 + \left\| E^c_{s \rightarrow t} \right\|_1 + \left\| P^o_{s \rightarrow t} - P^o_t \right\|_1 + \left\| E^o_{s \rightarrow t} - E^o_t \right\|_1,
\end{equation}
\right)$$


+ **$\mathcal{L}_{r}$ Reconsturction Loss**
    + source와 target 이미지가 동일한 경우 fidelity 보장
    + 동일 identity일 확률이 0.3인 source-target pair를 무작위 샘플링하여 사용

$$\left(
\mathcal{L}_r = \begin{cases}
    \left\lVert I^c_{s \to t} - I^c_t \right\rVert_1 + \left\lVert I^o_{s \to t} - I_t \right\rVert_1 & \text{if  } \text{id}(I_s) = \text{id}(I_t), \\
    0 & \text{otherwise}.
\end{cases}
\right)$$

+ **$\mathcal{L}_{g}$ Adversarial Loss**
    + 생성 이미지의 시각적 품질 향상
    + ${L}_{\text{adv}}(D(I^{o}_{s \to t}))$ : original space의 결과 이미지가 discriminator에게 real처럼 보이도록 
    <br>: swapped face의 realism ↑
    <br>: 조명/배경/텍스처 흐름의 일관성 ↑
    + ${L}_{\text{adv}}(D(I^{c}_{s \to t}))$ : canonical space에서 만들어지는 얼굴도 현실적인 canonical representation처럼 보이도록
    <br>: 중립적인 pose/expression을 가지는 canonical space의 얼굴을 부드럽고 왜곡되지 않게 표현

$$\left(
\mathcal{L}_{g} = \mathcal{L}_{\text{adv}}(D(I^{o}_{s \to t})) + \mathcal{L}_{\text{adv}}(D(I^{c}_{s \to t}))
\right)$$



 블렌딩 영역의 효과적인 비지도 학습을 가능하게 하여, 네트워크가 얼굴 스왑을 위한 적절한 경계를 자동으로 결정할 수 있게 하지만, 지나치게 날카로운 전환은 이러한 경계를 따라 아티팩트를 도입할 수 있음을 관찰했습니다




+ **$\mathcal{L}_{m}$ Mask Regularization Loss**
    + 자연스러운 마스크의 생성
    + ${L}_{tv}(A)$ : Total Variation Loss
        + $Ltv​(A)=i,j∑​∣Ai,j​−Ai+1,j​∣+∣Ai,j​−Ai,j+1​∣$
        + 마스크 경계 부분을 smoothing, 좌우상하 픽셀간 차이 최소화
    + $∥A−A_{GT}∥_1$ : L1 Reconstruction Loss
        + predicted mask를 GT 마스크와 동일하게
    
$$\left(
\mathcal{L}_m = \mathcal{L}_{tv}(A) + \|A - A_{GT}\|_1
\right)$$


+ **$\mathcal{L}_{total}$ Total Loss**
    + 앞의 Loss에 가중치를 적용하여 합산
    <br> $λid = λr = 10, λp = λmo = 5, λm = 1$
    + 모션 일관성과 identity consistency 확보

$$\left(
\mathcal{L}_{\text{total}} = \lambda_{\text{id}} \mathcal{L}_{\text{id}} + \lambda_{p} \mathcal{L}_{p} + \lambda_{\text{mo}} \mathcal{L}_{\text{mo}} + \lambda_{r} \mathcal{L}_{r} + \mathcal{L}_{g} + \lambda_{m} \mathcal{L}_{m}
\right)$$


## 4. Metrics of Video Face Swapping
+ 전통적인 VFS 평가 방법
    + ID 유사도/검색, expression 정확도, FID
    + temporal consistency, audio-lip 동기화에 제한적
+ 세분화된 평가 metric 제안 - 눈/입 영역에 대한 추가 측정 방법
    1) Eye
        + Gaze Estimation
        + Eye Eye Aspect Ratio ([EAR](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf))
    2) Lip Sync
    <br>: talking head synthesis task에서 LSE-D/LSE-C 사용
        + Lip Sync Error-Distance (LSE-D)
        <br>: 입 랜드마크와 audio에 따른 입 위치값에서 벗어난 평균 편차
        + Lip Sync Error-Confidence (LSE-C) 
        <br>: Lip Sync 예측 confidence

## 5. Experiment
+ 학습 데이터셋
    + VGGFace 데이터셋 
        + width < 130px 이미지 930K개
        + 512x512 크기로 resize
    
    + 참고. [FF++ dataset](https://github.com/ondyari/FaceForensics)
    <br>: 원본 영상 - 조작 영상(4종류) pair로 이루어진 1000개 video (1280x720)
    <br>: 딥페이크·조작 영상 검출 목적

+ 결과
    + ID 유사도 ↑
    + 타 알고리즘 대비
        + pose 오차 ↓
        + 표정 오차 ↓
<br>
<img width="404" height="542" alt="Image" src="https://github.com/user-attachments/assets/10c944d2-effc-4966-849d-773a88e6d04c" />
    + warping / masking / refinement 유무에 따른 결과 비교
<br>
<img width="404" height="542" alt="Image" src="https://github.com/user-attachments/assets/10c944d2-effc-4966-849d-773a88e6d04c" />