# Review
|Index|Title|Brief|
|----|----|----|
|1|[Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/pdf/2001.05566.pdf)||

# Challenges
|Index|Title|Brief|
|----|----|----|
|1|[Recurrent Saliency Transformation Network: Incorporating Multi-Stage Visual Cues for Small Organ Segmentation](https://arxiv.org/pdf/1709.04518.pdf)|Pancrease segmentation|

# High-resolution segmentation

|Index|Title|Brief|
|----|----|----|
|1|[Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images](https://arxiv.org/abs/1905.06368)|CVPR 2019 oral|
|2|[Label Super-Resolution Networks](https://openreview.net/pdf?id=rkxwShA9Ym)|ICLR 2019|
|3|[ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://hszhao.github.io/papers/eccv18_icnet.pdf)|ECC 2018|
|4|[CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement](http://hkchengad.student.ust.hk/CascadePSP/CascadePSP.pdf) ([code](https://github.com/hkchengrex/CascadePSP))|CVPR 2020|
|5|[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf)|PAMI|
|6|[SAN: Scale-Aware Network for Semantic Segmentation of High-Resolution Aerial Images](https://arxiv.org/pdf/1907.03089.pdf)|IEEE GEOSCIENCE AND REMOTE SENSING LETTERS|
|7|[Gated Convolutional Neural Network for Semantic Segmentation in High-Resolution Images](https://www.mdpi.com/2072-4292/9/5/446/htm)|Remote sensing|

# Unsupervised learning for segmentation
|Index|Title|Brief|
|----|----|----|
|1|[Medical Image Segmentation via Unsupervised Convolutional Neural Network](https://arxiv.org/pdf/2001.10155.pdf)|MIDL 2020|
|2|[Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering](https://arxiv.org/pdf/2007.09990.pdf)|IEEE TIP 2020|
|3|[Improving Robustness of Deep Learning Based Knee MRI Segmentation: Mixup and Adversarial Domain Adaptation](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Panfilov_Improving_Robustness_of_Deep_Learning_Based_Knee_MRI_Segmentation_Mixup_ICCVW_2019_paper.pdf)|ICCV 2019 膝关节分割，细读|
|3-r1|[Learning to Adapt Structured Output Space for Semantic Segmentation](https://arxiv.org/pdf/1802.10349.pdf)|文献２的引用文献，介绍了UDA的用法。|
|4|[Improving Data Augmentation for Medical Image Segmentation](https://openreview.net/references/pdf?id=B1-9HbnxX)|这篇文章提到用Mixup来做data augmentation。alpha=0.4时效果最好。|
|5|[UNSUPERVISED IMAGE SEGMENTATION BY BACKPROPAGATION](https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf)||
|6|[Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Weakly_Supervised_Learning_of_Instance_Segmentation_With_Inter-Pixel_Relations_CVPR_2019_paper.pdf)|CVPR 2019|
|7|[Digging Into Pseudo Label: A Low-Budget Approach for Semi-Supervised Semantic Segmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9003388)|Pseudo labelling for segmentation。提到了分割标注比分类任务繁重N倍。|
|8|[Self-Loop Uncertainty: A Novel Pseudo-Label for Semi-Supervised Medical Image Segmentation](https://arxiv.org/pdf/2007.09854.pdf)||

# Segmentation loss function
|Index|Title|Brief|
|----|----|----|
|1|[Correlation Maximized Structural Similarity Loss for Semantic Segmentation](https://arxiv.org/pdf/1910.08711.pdf)|这个loss function值得深入研究。|
|2|[On the Mathematical Properties of the Structural Similarity Index](https://ece.uwaterloo.ca/~z70wang/publications/TIP_SSIM_MathProperties.pdf)|这篇论文分析了ssim 的数学特性|
|3|[Adaptive Affinity Fields for Semantic Segmentation](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jyh-Jing_Hwang_Adaptive_Affinity_Field_ECCV_2018_paper.pdf)|引入了AAF的概念用于分割，需结合ssim一起理解。|

