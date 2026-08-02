# a bitter history for 3d detection / 1 pointpillar

> 一根柱子，如何把三维检测送进二维卷积的黄金时代

交互式配图版本：[PointPillars Geometry Lab](https://hova88.github.io/pointpillars-explained/)

---

## 写在前面

2019 年 3 月，Rich Sutton 写下了那篇后来被反复引用的短文 [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)。它的核心观察并不复杂：从足够长的时间尺度看，能够持续利用更多计算与数据的一般方法，往往会超过依赖人类经验、领域结构和精巧规则的专用系统。

同一年，PointPillars 出现在 CVPR。

把这两个事件放在一起看很有意思。PointPillars 一方面是 Bitter Lesson 的受益者：它用可学习的 PointNet 编码器替换了人工设计的 BEV 统计特征；另一方面，它又充满了非常具体的人类选择：规则网格、固定 pillar 尺寸、点数上限、最大池化、anchor 模板、IoU 阈值、方向分类和 NMS。

所以这篇文章并不打算把 PointPillars 简化为“一个过时的 anchor-based 检测器”，也不打算把后来的所有进步归结为“网络更大、数据更多”。更值得追问的是：

1. 2018—2019 年的 3D 检测到底卡在哪里？
2. PointPillars 为什么恰好选择了 pillar，而不是 point、voxel 或手工 BEV？
3. 每一个看似琐碎的操作，究竟保存了什么，又丢失了什么？
4. 从今天回看，哪些设计是 Bitter Lesson 迟早要冲走的脚手架？
5. 哪些设计反而证明，好的几何归纳偏置并不等于失败的人类知识？

本文主要依据 [PointPillars 原论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf)。文中的 nuScenes 场景只用于解释几何过程；原论文的实验结论来自 KITTI，二者的指标、类别、坐标范围和评测协议不能混用。

---

## 一、2019 年前夜：点云为什么不能直接当图片处理

一帧 LiDAR 点云可以写成一个无序集合：

$$
\mathcal P=\{p_i\}_{i=1}^{M},\qquad p_i=(x_i,y_i,z_i,r_i)
$$

其中 `r` 通常表示反射强度。这里最重要的不是多了一个 `z`，而是点云同时具有三种让 CNN 不舒服的性质：

- **无序**：交换两个点在数组中的位置，不应该改变场景语义。
- **稀疏**：激光只采到物体表面的一部分，绝大多数三维空间没有测量。
- **密度不均匀**：近处密、远处疏；遮挡、入射角、传感器线束与物体材质都会改变采样密度。

图像卷积默认输入位于规则的二维格点上。相邻像素有稳定的拓扑关系，数据又足够稠密，GPU 可以大批量地执行同一种计算。点云没有天然的行列索引；如果把三维空间全部体素化，又会得到一个巨大而几乎全空的张量。

早期方法大致在三条路之间选择：

### 1. 人工 BEV 编码

[MV3D](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf)、[PIXOR](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.html) 等方法把地面划成二维网格，再为每个格子计算高度、密度、强度等固定统计量。

优点是快，而且 BEV 对自动驾驶很自然：车辆在俯视图中的物理尺度相对稳定，遮挡也远少于透视图。缺点是统计量由人提前决定，网络只能学习如何使用这些特征，不能学习“应该测量什么”。

### 2. 直接学习点集

[PointNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html) 给出了一个关键答案：对每个点使用共享网络，再通过对称函数聚合，就能处理无序集合。

但车载 LiDAR 一帧可能有数万到十万点。对整个场景直接运行复杂点邻域查询，在当时的硬件与算子条件下并不便宜。

### 3. 学习 voxel 表示

[VoxelNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.html) 把三维空间划分为 voxel，在每个 voxel 内运行可学习的 VFE，再用 3D 卷积聚合空间上下文。它把“手工特征”改造成“端到端特征”，但原版推理速度约为 4.4 Hz。随后 [SECOND](https://www.mdpi.com/1424-8220/18/10/3337) 用稀疏卷积大幅加速，3D 卷积仍然是显著成本。

PointPillars 的问题意识因此非常明确：

> 能不能保留 VoxelNet 的可学习局部编码，却彻底删掉昂贵的三维卷积？

它的答案不是发明更复杂的三维算子，而是取消 `z` 方向的分箱。

---

## 二、Pillar：一次有目的的维度坍缩

设检测范围为：

$$
[x_{min},x_{max})\times[y_{min},y_{max})\times[z_{min},z_{max})
$$

PointPillars 只在 `x-y` 平面划分网格。若 pillar 尺寸为 `(v_x,v_y)`，点的网格索引为：

$$
i_x=\left\lfloor\frac{x-x_{min}}{v_x}\right\rfloor,\qquad
i_y=\left\lfloor\frac{y-y_{min}}{v_y}\right\rfloor
$$

同一 `(i_x,i_y)` 中、整个高度范围内的点被放进同一根竖直柱子，这就是 pillar。

### 为什么要 floor？

`floor` 把连续坐标映射为确定的离散地址。常见实现采用半开区间：左边界属于该格，右边界属于下一格；达到 `x_max` 或 `y_max` 的点已经越界。

这不是数学细节。若训练、数据增强和部署代码的边界约定不一致，同一点可能被写入不同位置，甚至产生越界索引。

### 为什么不划分 z？

不划分 `z` 带来三件事：

1. 不再需要人为选择竖直 voxel 高度。
2. 三维稀疏网格退化为二维 BEV 网格，后续可以完全使用成熟的 2D CNN。
3. 垂直结构不再由显式邻接关系表示，必须被压进 pillar feature。

这既是 PointPillars 的速度来源，也是它最根本的信息损失。

### 分辨率的代价

小 pillar 提供更精细的横向定位，但会增加非空 pillar 数量和 pseudo-image 尺寸；大 pillar 更快，却把更多表面混在一起。原论文默认 XY 分辨率为 `0.16 m`，并通过 `0.12—0.28 m` 的实验展示速度—精度折中。更大的格子主要伤害行人和骑行者，小目标比汽车更依赖精细空间分辨率。

---

## 三、从一个点到 9 维 decorated feature

原始点为：

$$
p_i=(x_i,y_i,z_i,r_i)
$$

设同一 pillar 中的点均值为：

$$
\bar p=(\bar x,\bar y,\bar z)
$$

设该规则网格的固定 XY 中心为：

$$
c_p=(x_p,y_p)
$$

原论文构造的 9 维特征是：

$$
f_i=[x_i,y_i,z_i,r_i,
x_i-\bar x,y_i-\bar y,z_i-\bar z,
x_i-x_p,y_i-y_p]
$$

一些文章把 `x_c,y_c,z_c` 写得像“质心坐标”，容易造成误解。在 PointPillars 的语义中，它们指向点相对 pillar 内算术均值的偏移；`x_p,y_p` 则是相对固定网格中心的偏移。

这三组信息分别解决不同问题。

### 1. 原始坐标 `(x,y,z,r)`：我在哪里

绝对坐标保留全局位置、高度与量测强度。完全删除它们后，两个局部形状相同、但位于道路不同位置的 pillar 会更难区分。

### 2. Cluster offset：我相对这组点在哪里

$$
\Delta p_{cluster}=(x_i-\bar x,y_i-\bar y,z_i-\bar z)
$$

为什么不能只给网络一个均值 `(\bar x,\bar y,\bar z)`？因为均值对 pillar 中每个点都相同。它告诉网络“这一组点大致在哪里”，却不能描述每个点如何围绕它分布。

一条车门边缘、一个行人躯干和一根路杆，完全可能具有相近的均值。区别它们的是点相对均值的形状。

Cluster offset 具有局部平移不变性。如果整根 pillar 的所有点平移同一向量，`p_i-\bar p` 不变。但它也因此不能替代绝对坐标，所以 PointPillars 同时保留二者。

### 3. Pillar-center offset：量化后，我在格子里的哪里

$$
\Delta p_{pillar}=(x_i-x_p,y_i-y_p)
$$

两个点经过 floor 后可能拥有同一离散索引，但一个贴近格子左下角，另一个贴近右上角。若只保留 `(i_x,i_y)`，这种亚格点位置已经丢失。相对固定中心的偏移把一部分量化误差重新交给网络。

原论文消融中，这两个额外的 pillar-center decoration 带来约 `0.5 mAP` 提升。这个结果只说明论文实验设置中的整体变化，不应被解释成任意实现、任意数据集上删除它们都会固定下降同样数值。

### 三种参照系缺一不可吗？

并不是数学上缺一不可，而是它们形成互补：

- 原始坐标保存全局位置；
- cluster offset 强调局部形状；
- pillar-center offset 补偿 XY 量化。

现代实现有时加入时间、距离、额外强度或多 sweep 属性，因此输入可能不再是 9 维。9D 是原论文设计，不是“PointPillars”这个名字的永恒定义。

---

## 四、为什么必须限制 P 和 N

每根 pillar 的点数不同，每帧非空 pillar 数量也不同。GPU 喜欢规则批量张量，因此论文设置：

- 每帧最多 `P` 根非空 pillar；
- 每根最多 `N` 个点。

原论文默认 `P=12000`、`N=100`，得到：

$$
X\in\mathbb R^{D\times P\times N},\qquad D=9
$$

超过容量时随机采样，不足时补零，并用 mask 防止 padding 被当成真实点。

### 这一步换来了什么？

- 固定形状便于批处理、内存规划和 1×1 卷积。
- 只保存非空 pillar，避免一开始就分配完整点集结构。
- 编码计算与 `P×N` 上限绑定，延迟更可控。

### 又丢失了什么？

- 密集近处表面超过 `N` 时会被截断。
- 当非空 pillar 超过 `P` 时，一部分空间证据被舍弃。
- “先到先得”、随机采样或确定性采样会引入不同偏差。
- padding 的数值零与真实的零测量必须依赖 mask 区分。

今天的动态 voxelization、ragged tensor 与更成熟的稀疏算子，正在尝试减少这类固定容量带来的信息损失。但在 2019 年，规则张量是非常现实的 GPU 友好选择。

---

## 五、Pillar Feature Net：一层 PointNet 到底做了什么

对每个 decorated point 使用同一组参数：

$$
h_i=\operatorname{ReLU}(\operatorname{BN}(Wf_i+b))
$$

形状从：

$$
(D,P,N)\rightarrow(C,P,N)
$$

原论文设置 `C=64`。线性层可以实现为 1×1 卷积，因此所有点、所有 pillar 可以高效并行。

### 为什么共享权重？

第 17 个点或第 2000 根 pillar 没有固定语义。如果为每个位置学习独立参数，模型会依赖任意存储顺序，参数量也会随容量增长。共享变换表达的是：“用同一种局部规则观察每个点。”

### 为什么 BatchNorm？

不同 feature 的尺度差异很大：米制坐标、反射强度、局部偏移并不在同一数值范围。BatchNorm 稳定中间激活，并在训练时使用 batch 统计，在推理时使用累计统计。小 batch、padding mask 与实现中的轴布局都会影响它的实际行为。

### 为什么 ReLU？

ReLU 提供廉价非线性并制造稀疏激活。它会把负激活压成零；这并不等于负坐标信息必然消失，因为线性层可以在进入 ReLU 前学习正负方向的不同通道。

随后沿点轴 `N` 做逐通道最大池化：

$$
g_{c,p}=\max_{n}h_{c,p,n}
$$

得到：

$$
G\in\mathbb R^{C\times P}
$$

### Max pooling 为什么适合无序点集？

任意重排 pillar 内的点，逐通道最大值不变，因此网络满足点序置换不变性。每个通道还可以由不同点“获胜”：一个点触发高度边缘，另一个点触发反射强度模式。

### Max pooling 丢掉了什么？

- 哪个点赢得了通道之外的大部分点信息；
- 点的频率与密度；
- pillar 内更细的相对排列；
- 次大响应与多峰结构。

Mean pooling 保留平均趋势，却对点密度敏感；sum pooling 更受点数影响；attention 更灵活，但更昂贵，也不自动解决截断问题。

PointPillars 只使用一层简化 PointNet。论文报告，相比 VoxelNet 的两层顺序 PointNet，这一选择节省约 `2.5 ms`。它不是追求最强局部建模，而是在问：一层是否已经足够把信息交给后面的空间 CNN？

---

## 六、Scatter：不是投影，而是恢复地址

PFN 输出的是紧凑列表：

$$
G\in\mathbb R^{C\times P}
$$

这里 `p` 只是“第 p 个非空 pillar”，不代表空间位置。预处理阶段还保留了坐标表：

$$
K[p]=(i_y,i_x)
$$

Scatter 执行确定性写入：

$$
I[:,K[p]_y,K[p]_x]\leftarrow G[:,p]
$$

最终得到：

$$
I\in\mathbb R^{C\times H\times W}
$$

没有非空 pillar 的位置被写成完整的零向量。

这一步没有插值、平均、卷积或可学习参数。它只是在紧凑列表与二维网格之间恢复映射关系。只要 feature row 与 coordinate row 成对移动，重新排列 `p` 的顺序不会改变最终 pseudo-image。

### Pseudo-image 到底是稀疏还是稠密？

最准确的说法是：

> 它在数值上稀疏，在原版 PointPillars 中却以稠密 `(C,H,W)` 张量存储和处理。

原论文估计典型 KITTI 设置中约有 97% pillar 为空，但 scatter 之后仍然使用普通 dense 2D CNN。

普通卷积会让非零响应传播到邻近零位置。这不是简单的“真实特征被稀释”，而是学习上下文：一根 pillar 的证据应该影响附近位置的检测判断。后来大量方法采用 sparse 或 submanifold convolution 以避免空位置计算，但保持稀疏也可能限制相邻活跃区域的信息交换。

因此，sparse convolution 是后来的重要替代路线，不是原版 PointPillars backbone 的组成部分。

---

## 七、2D backbone：PointPillars 真正押注的地方

PointPillars 的重要判断不是“二维比三维更真实”，而是：

> 把垂直信息先编码进 channel，剩下的大尺度空间推理交给高度优化的 2D CNN。

论文把 top-down block 写成 `Block(S,L,F)`：

- `S`：相对原 pseudo-image 的总 stride；
- `L`：3×3 Conv2D 层数；
- `F`：输出通道数。

汽车网络使用：

| Block | 总 stride | 卷积层数 | 通道数 |
|---|---:|---:|---:|
| B1 | 2 | 4 | 64 |
| B2 | 4 | 6 | 128 |
| B3 | 8 | 6 | 256 |

每层卷积后接 BatchNorm 与 ReLU。第一层负责降采样，后续层 stride 为 1。

更深的层分辨率更低，但感受野更大；浅层更容易保留精确位置。检测既需要“这里有一条细边缘”，也需要“附近整体像一辆车”。

随后三个尺度分别经过 transposed convolution：

- B1：`×1`，输出 128 channel；
- B2：`×2`，输出 128 channel；
- B3：`×4`，输出 128 channel。

它们被对齐到 stride 2，再沿 channel 维拼接：

$$
128+128+128=384=6C
$$

这是一种简洁的多尺度融合：高分辨率细节与低分辨率上下文在同一 detection head 前相遇。

Transposed convolution 并不会恢复降采样时丢失的原始细节，它学习的是如何把粗尺度 feature 映射回较密网格。插值加卷积也可以完成类似任务，并可能减少棋盘格伪影；这是通用架构选择，不是 PointPillars 特有问题。

---

## 八、SSD head：把二维位置变成三维盒子

PointPillars 采用单阶段 SSD 风格 detection head。对每个 BEV 位置、每个 anchor，网络预测：

1. 类别分数；
2. 7 个 box residual；
3. 离散方向类别。

若每个位置有 `A` 个 anchors、`K` 个类别，则典型输出通道数为：

- classification：`A×K`；
- box regression：`A×7`；
- direction：`A×2`。

具体实现可能使用独立类别网络、背景通道或不同的 class grouping，因此 shape 需要结合配置解释。

### Anchor 表达了什么？

一个 anchor 是先验盒子：

$$
a=(x_a,y_a,z_a,w_a,l_a,h_a,\theta_a)
$$

原论文为每个类别预设尺寸和 z 中心，并在 `0°` 与 `90°` 两个方向平铺。它把道路对象通常具有有限尺寸分布这一经验写进搜索空间。

好处是回归只需预测相对修正；坏处是 anchor 尺寸、数量、方向、类别模板和匹配阈值都需要配置。更多 anchors 可能提高召回，也会增加分类不平衡、输出通道和 NMS 成本。

### Ground-truth matching

论文使用 BEV 2D IoU 匹配 anchor 与 GT，不用高度和 z 参与匹配。匹配后，高度和 elevation 仍作为回归目标。

- 与某个 GT 的最佳 anchor 或超过正阈值：positive；
- 低于负阈值：negative；
- 中间区域：ignored。

Positive 同时学习分类、定位和方向；negative 只学习背景分类；ignored 不贡献 loss。

这是监督信号的入口。阈值不是后处理细节，它决定网络训练时“看见了哪些正例”。

---

## 九、Box coding：为什么不直接回归米制坐标

设 anchor 与 GT 均为 `(x,y,z,w,l,h,θ)`，并定义：

$$
d_a=\sqrt{w_a^2+l_a^2}
$$

PointPillars/SECOND 风格的目标为：

$$
\Delta x=\frac{x_{gt}-x_a}{d_a},\quad
\Delta y=\frac{y_{gt}-y_a}{d_a},\quad
\Delta z=\frac{z_{gt}-z_a}{h_a}
$$

$$
\Delta w=\log\frac{w_{gt}}{w_a},\quad
\Delta l=\log\frac{l_{gt}}{l_a},\quad
\Delta h=\log\frac{h_{gt}}{h_a}
$$

论文将角度项写为：

$$
\Delta\theta=\sin(\theta_{gt}-\theta_a)
$$

不同代码库可能直接回归角差，或采用其他周期编码，必须区分论文公式和实现约定。

### 为什么中心偏移要归一化？

同样 0.5 m 的误差，对行人和卡车不是同一相对尺度。用 anchor 对角线和高度归一化，使目标更接近无量纲，改善不同尺寸类别之间的数值条件。

### 为什么尺寸使用 log ratio？

尺寸天然为正。`log(w_gt/w_a)` 把乘性变化转为加性残差：预测零意味着尺寸等于 anchor，正负值分别表示放大和缩小。

### 为什么还需要方向分类？

角度回归存在周期性。一个长方体旋转 180° 后几何重叠几乎相同，但车辆朝向相反。论文使用离散方向 softmax 帮助恢复 heading，解决单纯角度定位 loss 无法区分的翻转。

---

## 十、Loss：哪些候选真正推动了学习

### Localization loss

$$
\mathcal L_{loc}=\sum_{b\in(x,y,z,w,l,h,\theta)}\operatorname{SmoothL1}(\Delta b)
$$

Smooth L1 在零附近近似二次函数，对小误差平滑；误差较大时转为线性增长，避免 L2 对离群值过度敏感。

### Classification loss

单阶段 detector 中，背景 anchors 远多于正例。论文使用 focal loss：

$$
\mathcal L_{cls}=-\alpha(1-p_t)^\gamma\log p_t
$$

并采用 `α=0.25, γ=2`。当样本已经很容易时，`(1-p_t)^γ` 会压低它的权重，使优化更关注难例。

### Direction loss

方向分箱使用 softmax classification loss：

$$
\mathcal L_{dir}
$$

### 总损失

$$
\mathcal L=\frac{1}{N_{pos}}
(2\mathcal L_{loc}+1\mathcal L_{cls}+0.2\mathcal L_{dir})
$$

`N_pos` 归一化让不同 batch 中正例数量变化不至于直接改变整体梯度规模。权重 `2,1,0.2` 是论文配置，不是从第一原则推导出的普适常数。

---

## 十一、数据增强：模型之外的半个算法

PointPillars 沿用了大量 SECOND 时代形成的增强策略。

### Ground-truth database sampling

先从训练集建立对象数据库：保存 GT box 以及盒内点。训练时把采样对象插入当前点云并更新 box。

原论文每帧随机采样 `15,0,8` 个汽车、行人、骑行者对象。这是特定 KITTI 设置下的结果，不应原样搬到 nuScenes。

GT sampling 改善稀有前景数量，却隐含一个强假设：只要碰撞检查与几何变换合理，把一个对象从场景 A 放进场景 B 仍然近似真实。错误的地面高度、遮挡关系、强度分布或上下文会制造不可能场景。

### Per-box augmentation

每个 GT box 及其内部点共同旋转和平移。若只移动 box 不移动点，监督立刻失效。

### Global augmentation

点云和所有 boxes 一起执行镜像、旋转、缩放与平移。联合变换保持几何关系，模拟姿态、尺度和定位噪声。

这些增强是 PointPillars 强结果的重要组成部分，却不是 pillar encoder 本身。论文甚至观察到，更激进的 per-box augmentation 会伤害行人，ground-truth sampling 已经减少了对它的需要。

这提醒我们：比较 detector 时，所谓“架构提升”可能混合了采样、增强、优化器、anchor 与 loss 的变化。

---

## 十二、推理：从 logits 回到同一帧点云

推理时，网络产生大量 dense candidates：

1. 把 logits 转成类别分数；
2. 用预测 residual 解码 anchor；
3. 过滤低置信度候选；
4. 按分数排序；
5. 执行 NMS；
6. 把保留 box 渲染回 LiDAR 坐标系。

解码是编码的逆过程，例如：

$$
x=\hat\Delta x\,d_a+x_a,\qquad
w=\exp(\hat\Delta w)w_a
$$

原论文使用 IoU 阈值 `0.5` 的 axis-aligned NMS。作者报告它与 rotated NMS 表现接近但更快。这又是一个非常 PointPillars 的决定：不是追求几何上最完美的算子，而是选择足够好且硬件友好的近似。

NMS 是贪心规则，不理解“这是两辆靠得很近的车”还是“同一辆车的两个候选”。在拥挤场景中，阈值过低会误删真实对象，过高会保留重复框。

最终结果仍然必须回到最初的物理场景：类别、分数、中心、尺寸与 yaw 都在同一坐标系中解释。没有这一步，前面的 pseudo-image 只是中间计算媒介。

论文给出的汽车网络总推理时间约 `16.2 ms`：其中 pillar 组织与 decoration、GPU 上传、PFN、scatter、backbone/head、CPU NMS 各自占据不同部分。最终 TensorRT 优化相比 PyTorch pipeline 带来约 `45.5%` 加速。62 Hz 不是抽象网络 FLOPs 的结果，而是算法与实现共同设计的产物。

---

## 十三、PointPillars 究竟丢掉了什么

### 1. XY 量化

连续坐标被绑定到离散格子。Pillar-center offset 可以补偿格内位置，却不能恢复跨格边界的连续邻接。

### 2. Z 邻接

不划分高度让计算变快，也让桥面、车辆顶部、地面和悬空结构在同一 XY 柱内竞争有限 channel。

### 3. 固定 P/N 与随机采样

容量上限让延迟可控，却会优先丢弃最密集区域中的点。

### 4. Max pooling

每个 channel 只保留最大响应。它擅长回答“是否存在这种局部模式”，不擅长回答“它出现了多少次、如何分布”。

### 5. Dense pseudo-image

点云天然稀疏，原版却在 scatter 后对所有 BEV 位置运行 dense CNN。它换来了成熟、高吞吐的 2D 算子，同时在空区域花费计算。

### 6. Anchor 与 NMS

模板尺寸、方向、IoU 阈值和 suppression 都是人为定义的搜索与去重规则。它们有效，但配置空间庞大，跨数据集迁移时经常需要重调。

---

## 十四、从今天回看：哪些是 Bitter Lesson？

先给出我的结论：

> PointPillars 不是 Bitter Lesson 的反例，也不是它的纯粹胜利。它是一次把“学习”推进到当时计算预算所允许边界的工程协定。

### 1. 被 Bitter Lesson 推动的部分：从固定统计量到 learned encoder

PointPillars 最明确的历史进步，是拒绝把每个格子永久压缩成几个人工统计量。

它保留坐标、反射强度和少量几何参照，再让共享网络学习什么局部模式重要。原论文的对照实验显示，learned encoders 在不同 pillar 分辨率下普遍优于固定编码；格子变大时差距更明显，因为固定统计量更难概括复杂点集。

这正符合 Bitter Lesson：不要把最终表示写死，让系统通过数据学习。

### 2. Anchor：典型的临时脚手架

Anchor 把对象尺寸、方向和匹配规则写入模型。2019 年它带来了稳定训练与成熟 SSD 工具链，绝不是无意义设计。

但 [CenterPoint](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html) 后来把对象视为中心点，通过 Gaussian heatmap 找中心，再回归尺寸、朝向与速度。它移除了 class-specific anchor 尺寸和方向枚举，训练目标也更接近对象本身。

从这个角度看，anchor 是 Bitter Lesson 容易冲走的一环：它把人对“应该在哪里搜索、以什么形状搜索”的理解预先编码得太具体。

但 CenterPoint 也不是完全没有人类先验。Gaussian 半径、BEV 网格、中心表示和回归 head 仍是设计。历史不是从“人工”跳到“纯学习”，而是不断把学习边界向外推。

### 3. 固定容量和 max pooling：计算条件留下的形状

`P×N`、随机采样与 max pooling 极其适合当时的 GPU，却强迫可变点集进入固定模具。

后来的动态 voxelization、集合注意力和稀疏网络试图处理任意数量局部元素。例如 voxel transformer 类方法允许更灵活的长程交互；它们通常也更依赖计算规模、数据量和成熟 kernel。

这部分很像 Bitter Lesson：随着算力和基础设施改善，更一般的聚合方式会逐步蚕食人为容量与单一 max 的必要性。

### 4. Dense BEV 不是简单的历史错误

原版 PointPillars 把 97% 左右为空的空间变成 dense pseudo-image，再运行普通 2D CNN。从今天看似浪费，但当时 dense Conv2D 的硬件效率可能胜过理论 FLOPs 更少、实际 kernel 不成熟的稀疏方案。

[VoxelNeXt](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_VoxelNeXt_Fully_Sparse_VoxelNet_for_3D_Object_Detection_and_Tracking_CVPR_2023_paper.html) 进一步展示了 fully sparse detector：不必 sparse-to-dense，不依赖 dense anchor/center proxy，甚至不需要 NMS。这里确实出现了一条更符合输入稀疏性、也更一般的路径。

但这并不说明 2019 年使用 dense 2D CNN 是错误。PointPillars 的关键能力正是利用当时最成熟、最可扩展的计算原语。Bitter Lesson 讲的是能够利用计算的通用方法；PointPillars 对 2D CNN 的押注，本身就符合这个精神。

### 5. Pillar 这个归纳偏置经受住了吗？

相当程度上，经受住了。

[PillarNeXt](https://openaccess.thecvf.com/content/CVPR2023/html/Li_PillarNeXt_Rethinking_Network_Designs_for_3D_Object_Detection_in_LiDAR_CVPR_2023_paper.html) 的观察很有意思：最简单的 pillar 模型在速度—精度上仍然很有竞争力，扩大感受野、更新 backbone 和训练设计就能获得强结果。这甚至挑战了“必须越来越精细地建模局部 3D 几何”这一常见直觉。

为什么？因为道路场景有强烈的地面结构，检测目标通常关心 BEV 中的位置、尺度和朝向。把垂直维压入 channel 并不是随意丢信息，而是与任务对称性相匹配的表示选择。

好的归纳偏置并不会自动违反 Bitter Lesson。真正危险的是无法随数据和计算扩展、又把错误假设锁死的知识。Pillar 把搜索空间从 3D 降到 2D，同时仍允许内部 feature 学习；它恰好是一种相对可扩展的偏置。

### 6. 数据增强与后处理：仍然 bitter 的角落

Ground-truth sampling、class-specific 阈值、方向 bin、score threshold 和 NMS 组成了大量非网络逻辑。它们通常对结果至关重要，却难以通过端到端目标共同优化。

这部分最符合“苦涩”：许多工程时间花在规则之间的耦合上，而规模扩大并不会自动让这些规则变好。anchor-free、query-based、set prediction 和 post-processing-free detector 的吸引力，很大程度上来自减少这些接口。

---

## 十五、PointPillars 真正留下的东西

如果只记住“把点云切成柱子，再做 2D CNN”，会低估这篇论文。

PointPillars 真正留下的是一种研究方法：

### 1. 把表示问题与硬件问题一起考虑

它没有追求最完整的三维表示，而是寻找能被 GPU 高效执行、同时保留足够检测信息的表示。

### 2. 在正确位置使用学习

它没有让网络重新发明坐标系，也没有把局部描述完全写死。规则网格负责地址，可学习 PFN 负责内容。

### 3. 让速度成为算法指标，而不是部署后的补丁

单层 PFN、64 channel、较窄 upsample feature、dense 2D conv、axis-aligned NMS 与 TensorRT 都服务于同一个目标：实时完整系统，而不是单个模块的漂亮 FLOPs。

### 4. 接受有意识的信息损失

工程系统不可能保存全部信息。关键不是“有没有损失”，而是：丢掉的信息是否与任务和预算相匹配，剩余信息是否还能被下游网络利用。

---

## 结语：这根 pillar 是一场暂时停战

2019 年的 PointPillars 位于两股力量之间。

一边是三维世界的复杂性：无序、稀疏、非均匀、旋转、遮挡和可变密度。

另一边是当时最强大的计算机器：喜欢规则张量、批量矩阵乘法和高度优化的二维卷积。

Pillar 是双方达成的一场停战。它没有完整保留三维结构，也没有退回人工统计量；它用最少的几何脚手架，把尽可能多的表示学习交给网络，再把整个场景送入 2D CNN。

从今天看，anchor、固定容量、max pooling、dense head 和 NMS 都显得带有时代痕迹。它们是 Bitter Lesson 继续发生的地方。

但 PointPillars 最核心的决定——用一种简单、规则、可扩展的表示释放成熟计算原语——并没有被历史轻易否定。后来更强的 pillar 网络反而提醒我们：Bitter Lesson 不是“永远不要使用结构”，而是不要让结构阻止系统利用更多数据、计算和学习。

PointPillars 最值得纪念的，也许不是它曾经在 KITTI 上有多快，而是它提出了一个非常诚实的问题：

> 在有限算力下，三维检测真正需要保留多少个维度？

这个问题，到今天仍然没有过时。

---

## 参考资料

1. Richard S. Sutton, [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), 2019.
2. Alex H. Lang et al., [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.pdf), CVPR 2019.
3. Charles R. Qi et al., [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html), CVPR 2017.
4. Yin Zhou and Oncel Tuzel, [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.html), CVPR 2018.
5. Yan Yan, Yuxing Mao, and Bo Li, [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337), Sensors 2018.
6. Bin Yang, Wenjie Luo, and Raquel Urtasun, [PIXOR: Real-Time 3D Object Detection from Point Clouds](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.html), CVPR 2018.
7. Tianwei Yin, Xingyi Zhou, and Philipp Krähenbühl, [Center-Based 3D Object Detection and Tracking](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html), CVPR 2021.
8. Jinyu Li, Chenxu Luo, and Xiaodong Yang, [PillarNeXt: Rethinking Network Designs for 3D Object Detection in LiDAR Point Clouds](https://openaccess.thecvf.com/content/CVPR2023/html/Li_PillarNeXt_Rethinking_Network_Designs_for_3D_Object_Detection_in_LiDAR_CVPR_2023_paper.html), CVPR 2023.
9. Yukang Chen et al., [VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_VoxelNeXt_Fully_Sparse_VoxelNet_for_3D_Object_Detection_and_Tracking_CVPR_2023_paper.html), CVPR 2023.

---

## 发布备注

- 本文中的原版 PointPillars 参数与性能数字均对应论文的 KITTI 实验。
- 交互网站使用 nuScenes 样例帧解释几何过程，不用该帧声称模型精度。
- 网站最终章中的红框是官方 nuScenes GT；蓝框是确定性教学几何，不是训练模型输出。
- 若发布到 Medium，可将 LaTeX 公式导出为图片；知乎通常可直接使用公式编辑器重新录入。
