# [Study 10.도진경] Depth Anything 3: Recovering the Visual Space from Any Views

+ [Title] Depth Anything 3: Recovering the Visual Space from Any Views
+ [Publication] 2025.11 (ICLR 2026 Oral)
+ [Reference] [paper](https://arxiv.org/abs/2511.10647) | [github](https://github.com/ByteDance-Seed/depth-anything-3) | [project page](https://depth-anything-3.github.io/)

<br>
<p align='center'>
    <img width="381" height="297" alt="Image" src="https://github.com/user-attachments/assets/41744256-5648-4a2f-8f63-0b147fcf98ec" />
</p>

## Abstract
+ single plain transform 를 backbone으로 사용 (vanilla DINO encoder)
+ teacher-student 학습 기반 DA2 수준의 detail 및 generalization 

## 1. 개요
+ 기존 Monocular Depth, SfM, SLAM 등 각 task 별 특화된 복잡한 구조의 모델 - VGGT, DUSt3R
    + 대규모 학습 모델
+ single transformer 기반 입력 데이터(단일/다중 이미지, 비디오)로부터 3D 구조 복구
+ DINOv2 같은 pre-trained ViT를 backbone으로 사용
	+ input-adaptive cross-view self-attention -> 모든 시점의 정보 교환
	+ dual DPT head: depth, ray 출력 -> 다른 fusion param을 사용해서 동일 feature로부터 depth/ray를 동시에 출력
+ 학습
    + teacher-student paradigm 기반
    + 학습 데이터: real-world depth camera captures, 3D reconstruction, and synthetic data, where real-world depth may be of poor quality

<br>

## 2. Related Works
+ Multi-view visual geometry estimation.
    + 전통적 방식 (SfM, MVS)
     <br> : 특징점 검출 → 매칭 → 포즈 추정 → bundle adjust → Dense Depth. 
     <br> : 텍스처가 풍부한 곳에서 robust, 특징점이 적은 곳에서는 취약
    + E2E
     <br> : 이미지 pair 간 point feature > matching > pose/depth 추정 과정을 directly regress
+ Monocular depth estimation.
    + 실내/주행 환경 등 제약 환경에서의 supervised learning 위주
    + DiT, 대용량 학습 데이터셋 기반 monocular depth 추정
+ Feed-Forward Novel View Synthesis
    + NeRF - 장면 별 optimization 필요, 많은 연산량
    + 3D Gaussian (3DGS) - 실시간 렌더링 가능


<br>

## 3. Method
> 
### 3.1. Formulation
> · 3D 공간상 점 $P$를 2D 공간상 점 $p$로 변환/복구 

$$\left( P = R_i D_i (u,v) K^{-1}_i p + t_i\right)$$
+ 2D point $p = (u,v,1)^T$ 를 3D point $P=(X,Y,Z,1)^T$로 projection
    + $D_i(u,v)$: (u,v) 위치에서의 depth값
    + $R_i|t_i$: extrinsic param
    + $K_i$: intrinsic
    + $v$: 카메라 표현 파라미터
        + t: translation
        + q: quaternion
        + f: fov

> · Depth-ray representation
> : 회전 행렬 $R$ 대신, 각 픽셀이 뻗어 나가는 광선(Ray)를 예측
+ $P = t + D(u, v) · d$
    + 카메라 위치 t에서 시작해서, D 깊이만큼 d 방향으로 간다. -  world coordinate에서의 3D Point
    + 예측된 깊이 $d$, ray 맵 $M$을 곱하고 더하는 것만으로 3D point cloud 생성 가능 
    + R, K가 하나의 벡터로 표현되므로 각 요소를 복잡하게 계산할 필요 x
    ( p를 카메라 프레임으로 backproject & world로 rotate )

+ Ref.
    + $r=(t,d)$
        : 카메라 ray; $r$를 원점 $t$와 방향 $d$로 정의
    + $d = R K^{-1} p$
        + $R K^{-1} p$: world에서 p의 방향
            + $K^{-1} p$ : 카메라를 원점으로 한 3D 공간에서 p의 방향
            + $R$ : world에서 카메라의 rotation 값
    + $M ∈ R^{H×W ×6}$
        + dense ray map, 모든 픽셀에 대한 파라미터 저장
        + $M(:,:,:3)$ : 각 픽셀이 시작되는 ray의 원점(Origin) 정보
        + $M(:,:,3:)$ : 각 픽셀에 저장된 ray 방향

> · Deriving Camera Parameters from the Ray Map
> : Ray map으로부터 카메라 param$(t,R,K)$를 도출한다.

$$\left(
t_c = \frac{1}{H \times W} \sum_{h=1}^{H} \sum_{w=1}^{W} \mathrm{M}(h, w, : 3)
\right)$$
+ $t_c$: 카메라의 3d 위치
    + 각 픽셀 별 ray 원점의 평균값 (모든 픽셀이 같은 origin)
$$\left(
\mathbf{H}^* = \arg \min_{\|\mathbf{H}\|=1} \sum_{h=1}^{H} \sum_{w=1}^{W} \|\mathbf{H}\mathbf{p}_{h,w} \times \mathbf{M}(h, w, 3 : )\|.
\right)$$
+ $H$<br>: 호모그래피를 사용해서 $R$,$K$를 추정
    (R: 카메라 rotation, K: intrinsic param, H=RK)<br>: world <-> camera coordinate 간 변환. <br>: 위 수식상 오차를 최소화하는 H 도출 (RQ Decomposition 사용)
    + $d_I = K_I^{-1} p = p$ : canonical space 에서 ray 방향
    + $d_{cam} = K R d_I$ : camera coordinate 상의 ray 방향

> · Minimal prediction targets.
> : Depth와 Ray라는 최소한의 타겟만 사용, 빠르고 효율적인 연산
+ 타겟 최소화를 통한 Entanglement 방지
+ lightweight Camera Head ($\mathcal{D}_C$) 사용
    + ray map에서 camera pose 복구 시 연산을 줄이기 위해 사용
    + 트랜스포머는 camera token 기반 FOV(${f}$), 회전(${q}$), 이동(${t}$) 예측

> 회전 행렬을 직접 구하지 말고 레이를 예측한다.

<br>

### 3.2. Architecture
+ 효율성과 확장성을 극대화한 hybrid transformer 구조

> · Single transformer backbone


+ DINOv2와 같은 표준 ViT 하나만 사용
+ Token Rearranging
    + 력 토큰의 순서를 바꾸는 것만으로 여러 이미지를 동시에 처리
+ $L = Ls + Lg$   
    + $L_s$ : 각 이미지 내 self-attention, 한 이미지 특성 파악
    + $L_g$ : 모든 token에 대해 cross view / within-view attention 수행, 모든 이미지 간 관계 파악
    + $Ls : Lg = 2 : 1$

> · Camera condition injection

+ 카메라 정보(포즈, 내적 파라미터)가 있을 때와 없을 때를 모두 처리 가능
    + 카메라 정보가 있는 경우, mlp $Ec$를 통해 camera token 생성 $ci = Ec(fi, qi, ti)$

> · Dual-DPT head.  (Figure 3)

+ Depth, Ray 생성
    + 백본에서 나온 feature를 Reassembly modules에서 처리, 마지막 단계에서만 Depth/Ray로 나뉜다.
    + 각 branch 는 동일한 feature 공유
    + Fusion layers의 파라미터만 다르다.
+ 두 값이 동일 feature 기반으로 추정되어 높은 alignment, 낮은 연산량

> - "하나의 똑똑한 뇌(Backbone)가 카메라 정보를 참고하여(Condition), 한 번에 깊이와 광선 지도를 그려내는(Dual-DPT)" 구조
> - $L_s:L_g$ 비율 조절을 통해 성능과 속도의 최적점을 찾았다


### 3.3 Training
> · Teacher-student learning paradigm

+ Real World 데이터는 Depth 값이 비거나 노이즈가 많아 모델 학습 어려움.
    + teacher-student를 사용해서 real data의 구조 유지, teacher 모델이 가진 detail 표현 능력 적용
+ Teacher 모델: Synthetic data로만 학습된 monocular depth estimation
+ Pseudo-labeling : real-world 이미지의 depth 값을 Teacher 모델이 대신 예측
+ RANSAC Least Squares: Teacher가 만든 depth 맵을 real 데이터(Sparse/Noisy GT)의 스케일에 맞춰 정렬, real 데이터의 구조적 정보를 유지한다.

> · Training objectives


$$\left(
\mathcal{L} = \mathcal{L}_D(\hat{D}, D) + \mathcal{L}_M(\hat{R}, M) + \mathcal{L}_P(\hat{D} \odot d + t, P) + \beta \mathcal{L}_C(\hat{c}, v) + \alpha \mathcal{L}_{\text{grad}}(\hat{D}, D)
\right)$$
$$\left(
\mathcal{L}_{\mathrm{D}}(\hat{\mathbf{D}}, \mathbf{D}; D_c) = \frac{1}{Z_{\Omega}} \sum_{p \in \Omega} m_p \left( D_{c,p} |\hat{D}_p - D_p| - \lambda_c \log D_{c,p} \right),
\right)$$
+ $\mathcal{L}_D$ 
    + Depth Loss, 예측 깊이($\hat{\mathbf{D}}$)와 정답($\mathbf{D}$) 간 차이 최소화
    + Confidence($D_c$)를 사용, 확실한 영역에 높은 가중치
+ $\mathcal{L}_M$ 
    + Ray Map Loss, 예측 Ray Map($\hat{\mathbf{R}}$)과 정답($\mathbf{M}$) 간 차이 최소화
+ $\mathcal{L}_P$ 
    + Point Cloud Loss, 예측 Depth ($\mathbf{D}$)와 ray($\mathbf{d}$) 기반 변환된 3D point와 실제 3D 포인트($\mathbf{P}$) 간 차이 최소화
    + depth와 ray의 alignment를 맞춘다.

$$\left(
L_{\text{grad}}(\hat{D}, D) = ||\nabla_x \hat{D} - \nabla_x D||_1 + ||\nabla_y \hat{D} - \nabla_y D||_1,
\right)$$
+ $\mathcal{L}_{grad}$ 
    + Gradient Loss, depth의 변화량 
    + 물체의 edge 유지, 평평한 부분은 부드럽게 유지
+ $\mathcal{L}_C$ 
    + Camera Loss, 옵션으로 출력되는 카메라 포즈($\hat{\mathbf{c}}$)


### 3.4 Implementation Details
+ 학습 데이터셋
    + Synthesis Dataset(HyperSim, Objaverse 등)
    + Real-World Dataset (ScanNet++, ArkitScenes 등)
    + view point: 2~18장 이미지
    + 해상도: 504x504(default), 2:3, 3:4, 9:16 등 실제 사진 비율에 맞춰 다양하게 샘플링
+ 학습 환경: H100 GPU 128개, 200k step
+ Pose Conditioning: 학습 중 20% 확률로 카메라 포즈 정보 사용



## 4. Teacher-Student Learning
### 4.1 Constructing the Teacher Model
> Data scaling

+ DA2 대비 학습 데이터셋 양 증가 - Hypersim, GTA-SfM 등 20개가 넘는 합성 데이터셋을 사용
    + indoor/outdoor에서의 성능 향상
> Depth representation

+ DA2: scale-shift-invariant disparity(시차)를 예측
+ teacher model: scale-shift-invariant depth 를 예측
    + sfm, slam 등 depth를 추정하는 downstream task는 depth space에서 진행 <br> : disparity 예측 시 disparity->depth 변환 과정에서 오차 누적
    + linear depth 대신 exponential depth 예측 <br> : near camera 에서의 차이 인식 성능 확보 목적

> Training objectives

+ ROE alignment with the global–local loss 사용
    + surface normal + gradient + alignment로 Depth의 geometry 학습

+ **distance-weighted surface-normal loss**
+ 중심 pixel에서 4개의 neighbor point 선택, unnormalized normal ${n_i}$ 계산 <br> : noise나 특정 방향으로의 bias를 줄이기 위해
    + 다음 weight를 적용, 중심으로부터 멀수록 낮은 weight
        + $n_i = (p_i - p_c) \times (p_{i+1} - p_c)$  <br> : 중심 픽셀과 이웃 point로 계산된 i번째 local surface normal <br> : 중심 픽셀과 이웃 두 점으로 만든 삼각형의 법선 벡터, 값이 클수록 중심에서 멀다.
    
$$\left(
w_i = \sum_{j=0}^{4} \left\| n_j \right\| - \left\| n_i \right\|,
\right)$$
+ 4개의 인접 pixel과의 관계에서 구한 $w_i, n_i$를 사용하여 평균 surface normal 계산
$$\left(
n_m = \sum_{i=0}^{4} w_i \frac{n_i}{\| n_i \|},
\right)$$
$$\left(
\mathcal{L}_N = \mathcal{E}(\hat{n}_m, n_m) + \sum_{i=0}^1 \mathcal{E}(\hat{n}_i, n_i)
\right)$$
+ Normal Loss
    + $\mathcal{E}(\hat{n}_m, n_m)$: 평균 normal 비교, 전체 surface 방향
    + $\mathcal{E}(\hat{n}_i, n_i)$: 개별 normal 비교, local detail
$$\left(
\mathcal{L}_T = \alpha \mathcal{L}_{\text{grad}} + \mathcal{L}_{\text{gl}} + \mathcal{L}_{N} + \mathcal{L}_{\text{sky}} + \mathcal{L}_{\text{obj}}
\right)$$
+ 최종 loss
    + $L_{grad} = \|\nabla D - \nabla D_{gt}\|$ <br>: depth의 edge/형태 유지
    + $L_{gl}$: scale–shift invariant depth alignment (scale/shift 보정)
    + $L_{N}$: surface normal
    + $L_{sky}, L_{obj}$: mask loss, sky/object-only 데이터의 BG 에 대해서는 연산 진행 x
- depth 와 normal 모두를 맞추면 기하학적 구조의 일관성이 높아짐 <br> : surface가 자연스럽게 이어지고, stitching/3D recon 작업에 유리

### 4.2 Teaching Depth Anything 3

$$\left(
(\hat{s}, \hat{t}) = \operatorname{arg\,min}_{s>0, t} \sum_{p \in \Omega} m_p (s \tilde{\mathbf{D}}_p + t - \mathbf{D}_p)^2
\right)$$
+ scale/shift가 없는 depth($D$)를 → 실제 거리 스케일로 alignment
    + $\tilde{\mathbf{D}}$: teacher depth, 상대적 depth
    + $D$: noisy한 실제 depth - COLMAP,LiDAR 등에서 확보
    + $m_p$ : validity mask, depth가 존재하는 pixel만 사용
$$\left(
\quad \mathbf{D}^{T \to M} = \hat{s} \tilde{\mathbf{D}} + \hat{t}
\right)$$
+ monocular로 나온 상대적 거리를 실제 거리에 맞게 scaling/shift
///
- noisy한 실제 depth를 기준으로, teacher의 relative depth를 robust하게 scale–shift 정렬하여 metric depth supervision으로 사용
    + scale consistency
    + pose-depth 정렬
    + noisy한 real world에서의 안정성

### 4.3 Teaching Monocular Model
+ 라벨이 없는 이미지에 대해 student monocular model 학습
    + 상대적인 depth 추론
### 4.4 Teaching Metric Model
+ 절대 거리를 추론하는 student metric model 학습
    + Depth는 focal length에 따라 정해진다 - 모든 depth를 동일한 카메라 기준으로 정규화
    + $D' = f_c / f · D $
        + $f_c$: 표준 focal length
        + $f$: 실제 카메라의 focal length


## 5. Application: Feed-Forward 3D Gaussian Splattings
### 5.1 Pose-Conditioned Feed-Forward 3DGS
+ 카메라 포즈 정보가 주어졌을 때 3D 복원
    + GS-DPT head 사용 <br> : single transformer backbone을 사용해서 camera 공간 내 3D Gaussian 파라미터 추정

### 5.2 Pose-Adaptive Feed-Forward 3DGS
+ 카메라 포즈 정보가 없을 때 3D 복원
    + single transformer backbone을 사용, 상대적 Depth 예측
    + GS-DPT head에서 Camera space 3D 생성 및 camera pose 예측
    + World Space 변환
    