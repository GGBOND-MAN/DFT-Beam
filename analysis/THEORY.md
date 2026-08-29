# 理论设计：无 TTD 宽带近场 DFT 波束训练

约定：ULA 沿 y 轴，N 个阵元，阵元索引偏移 $\delta_n=(2n-N-1)/2$，间距 $d=\lambda_c/2=c/(2f_c)$
（相移器阵列，间距由中心频率固定，不随子载波改变）。空间角 $\theta=\sin\varphi\in[-1,1]$，
距离 $r$。OFDM 子载波 $f_m$，记 $\eta_m\triangleq f_m/f_c$，分数带宽 $\beta\triangleq B/f_c$，
故 $\eta_m\in[1-\beta/2,\;1+\beta/2]$。

## 1. 观测模型

近场球面波下第 $n$ 阵元到用户的距离 $r_n=\sqrt{r^2+\delta_n^2d^2-2r\theta\delta_n d}$，
Fresnel 展开 $r_n\approx r-\delta_n d\,\theta+\dfrac{\delta_n^2d^2(1-\theta^2)}{2r}$。

子载波 $m$ 的信道导向矢量与**频率无关**的 DFT 码字：

$$[\mathbf b_m(\theta,r)]_n=\tfrac1{\sqrt N}e^{-j2\pi f_m(r_n-r)/c},\qquad
[\mathbf a(\phi)]_n=\tfrac1{\sqrt N}e^{-j\pi\delta_n\phi}$$

码字在中心频率设计、所有子载波复用——这就是"无 TTD"的全部含义。波束增益

$$G_m(\phi;\theta,r)=\big|\mathbf b_m^H\mathbf a(\phi)\big|
=\tfrac1N\Big|\sum_n \exp\!\big(-j\pi\delta_n(\eta_m\theta+\phi)
+j\pi\eta_m\tfrac{\delta_n^2 d(1-\theta^2)}{2r}\big)\Big|$$

关键代数事实：因 $2\pi f_m d/c=\pi\eta_m$，**线性项与二次项被同一个 $\eta_m$ 提出**。

观测为幅度（不使用相位）：$z_{m,k}=|y_{m,k}|$，$y_{m,k}=A_m G_{m,k}e^{j\psi_{m,k}}+w$，
$w\sim\mathcal{CN}(0,\sigma^2)$，故 $z_{m,k}$ 服从 Rician 分布。

## 2. 命题 1（缩放映射）

$$\boxed{\;G_m(\phi;\theta,r)=G_c(\phi;\theta_m,r_m),\qquad
\theta_m=\eta_m\theta,\quad r_m=r\,\frac{1-\eta_m^2\theta^2}{\eta_m(1-\theta^2)}\;}$$

由匹配线性项 $\theta_m=\eta_m\theta$ 与二次项 $\dfrac{1-\theta_m^2}{r_m}=\eta_m\dfrac{1-\theta^2}{r}$ 得到。

**含义**：M 个子载波不是 M 个独立观测，而是在 $(\theta,r)$ 平面上一条**已知的一维曲线**
$\eta\mapsto(\theta_\eta,r_\eta)$ 上对同一个图样函数采样。频率维**不产生新的物理观测量**。

数值验证（`exp9`）：相对误差 0.04%–1.05%；忽略缩放则为 28%–107%。

## 3. 命题 2（图样特征）

由驻相条件 $\phi=-\eta_m\theta+\eta_m\delta_n d(1-\theta^2)/r$，$\delta_n\in[-N/2,N/2]$：

$$C_m=-\eta_m\theta,\qquad W_m=\eta_m\,\frac{Nd(1-\theta^2)}{r}$$

换算成 DFT 波束数（栅格间距 $2/N$）：

$$\boxed{\;W^{\rm beams}\approx\eta_m\,\frac{N^2d(1-\theta^2)}{2r}\;}$$

中心与宽度按**同一个** $\eta_m$ 缩放——这是 Pattern Zooming 在 ULA 上的形式，
也说明"不同频率给出不同 $\theta/r$ 灵敏度因而解耦"不成立。

数值验证：实测/预测比 0.68–1.14（差异来自半高阈值约定）。

## 4. 命题 3（栅格抖动）与采样定理

图样中心在带内的行程：

$$\boxed{\;\Delta^{\rm beams}=\big|C_{\max}-C_{\min}\big|\big/(2/N)=\frac{\beta|\theta|N}{2}\;}$$

设码本抽稀因子为 dec（每 dec 个 DFT 波束取一个）。窄带时图样上的采样间隔即 dec。
宽带时 M 条栅格以 $\Delta$ 为跨度交错，最坏采样间隔

$$\boxed{\;g=\max\Big(\mathrm{dec}-\Delta,\;\frac{\Delta}{M-1}\Big)\;}$$

（窄带是 $M=1,\Delta=0$ 的特例，$g=\mathrm{dec}$。）
训练成功要求 $g\lesssim W/2$，导频节省因子

$$\boxed{\;\text{saving}=\frac{\mathrm{dec}}{\max(g,1)}\;}$$

上限 dec，因为图样本身带限，采样再密也不可能优于全扫描。

**预测校验**（N=256, β=0.05, M=9）：
- θ=0.6, dec=8：$\Delta$=3.84, $g$=4.16, 预测 1.9×，实测 2×
- θ=0.6, dec=4：$\Delta$=3.84, $g$=0.48, 预测 4×，实测 4×

## 5. 三条边界（必须写进论文）

1. **抖动不创造距离信息。**距离信息来自宽度 $W$；$W\gtrsim1$ 波束是硬上限
   （$r\lesssim N^2d(1-\theta^2)/2\approx$ Rayleigh/2），频率不改变它。
   抖动只让稀疏栅格采到本来会被跨过去的宽度。
2. **正侧射盲点。**$\Delta\propto|\theta|$，$\theta\to0$ 时节省归零。
3. **前端上限。**纯 PS 的宽带阵列增益饱和在 $O(1/\beta)$ 且与 N 无关
   （β=0.05 时损失 5.3 dB）。这是数据传输端的代价，与训练端无关，但必须声明。

## 6. 估计器

参数 $(\theta,r)$，冗余参数为每子载波幅度 $\{A_m\}$（路径损耗随频率变化）与 $\sigma^2$。

Rician 对数似然 $\ell=\sum_{m,k}\log p(z_{m,k}\mid\theta,r,A_m,\sigma^2)$。
为避免 $\{A_m\}$ 的联合搜索，采用**幅度最小二乘的 profile 形式**：

$$(\hat\theta,\hat r)=\arg\max_{\theta,r}\;
\sum_m\frac{\big(\sum_k z_{m,k}G_{m,k}(\theta,r)\big)^2}{\sum_k G_{m,k}^2(\theta,r)}$$

$A_m$ 已闭式消去（$A_m^\star=\langle z_m,G_m\rangle/\|G_m\|^2$），
每子载波独立归一化，因此对频率相关增益免疫。
求解：$(\theta,1/r)$ 上的粗网格 + Nelder-Mead 精化。

注意这是 LS 而非精确 Rician MLE；高 SNR 下两者等价，低 SNR 下 LS 有偏。
论文中若要贴 CRB，需换成真正的 Rician MLE。

---

## 7. 实验验证结果（`analysis/estimator.py`）

### 7.1 无噪声全局可辨识性（exp9）

窄带全码本、窄带 dec=4、宽带 dec=4 三种配置下，真值均为 profile LS 代价函数的
**全局最大点**（搜索范围 r ∈ [5,300] m）。因此后续任何失败都是门限效应，
不是根本性歧义。

### 7.2 信噪比门限（exp10，θ=0.6, r=12 m）

| 方案 | 导频 | A0 | 峰值 SNR | RMSE_r | CRB_r | 比值 |
|---|---|---|---|---|---|---|
| 窄带全码本 | 256 | 30 | 22.9 dB | 0.380 | 0.400 | 0.9 |
| 窄带 dec=4 | 64 | 30 | 20.0 dB | 32.79 | 0.824 | **39.8** |
| 宽带 dec=4 | 64 | 30 | 14.0 dB | 13.90 | 0.760 | **18.3** |
| 窄带 dec=4 | 64 | 100 | 30.4 dB | 0.242 | 0.243 | 1.0 |
| 宽带 dec=4 | 64 | 100 | 24.5 dB | 0.262 | 0.220 | 1.2 |
| 窄带 dec=4 | 64 | 300 | 40.0 dB | 0.077 | 0.080 | 1.0 |
| 宽带 dec=4 | 64 | 300 | 34.0 dB | 0.076 | 0.073 | 1.0 |

门限之上估计器精确达到 CRB（比值 0.9–1.2），估计器设计正确。
门限之下稀疏码本崩溃。注意宽带在**低 6 dB 的峰值 SNR** 下崩得更轻。

### 7.3 误差分布（exp11，r=12 m，12 个角度 × 40 次）

A0=100（门限附近）：

| 方案 | 导频 | 中位 | p90 | 最大 | 可用率 |
|---|---|---|---|---|---|
| 窄带全码本 | 256 | 0.108 | 0.350 | 0.73 | 100% |
| 窄带 dec=4 | 64 | 0.628 | 29.41 | 48.0 | 71.5% |
| 宽带 dec=4 | 64 | 0.276 | 48.00 | 48.0 | 75.0% |

**此处频率维只改善中位数，不能挽救尾部**，CRB 预测的 4× 节省未兑现。

A0=300（门限之上）：

| 方案 | 导频 | 中位 | p90 | 最大 | 可用率 |
|---|---|---|---|---|---|
| 窄带全码本 | 256 | 0.027 | 0.078 | 0.18 | 100% |
| 窄带 dec=4 | 64 | 0.197 | 3.458 | 27.11 | 80.4% |
| **宽带 dec=4** | **64** | **0.065** | **0.734** | **2.29** | **100%** |

**此处主张成立**：同为 64 导频，宽带可用率 100% 对窄带 80.4%，
p90 压低 4.7×，最大误差压低 11.8×，中位改善 3×。
相对 256 导频全扫描，中位精度差 2.4×，但可用率同为 100%，开销与能量均为 1/4。

### 7.4 由此得到的结论

方向**成立，但必须附带信噪比条件**。CRB 是局部界，在稀疏码本下会
系统性高估可达性能——第 4 节的节省因子只在门限之上有效。论文必须包含
门限曲线，否则 CRB 表格会误导。
