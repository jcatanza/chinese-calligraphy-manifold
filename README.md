# chinese-calligraphy-manifold
 ![The Manifold River](the_manifold_river.jpg "River manifold")

 **Comparative Evaluation of Conditional Generative Models**

**for Chinese Character Synthesis and Data Augmentation:**

**C-VAE, C-GAN, and C-VAEGAN**

Joseph Catanzarite

Johns Hopkins University -- Whiting School of Engineering

EN.705.603 -- Introduction to Generative AI

*Spring 2026 -- Midpoint Draft*

# Abstract

Deep learning models for Chinese character recognition face a fundamental challenge: there are thousands of distinct character classes, each visually intricate, and collecting enough labeled examples of all of them is prohibitively expensive. This paper proposes a generative data augmentation strategy --- using machine learning models to *dream up* new synthetic training examples --- and asks which of three distinct generative architectures does it best. The three candidates are the Conditional Variational Autoencoder (C-VAE), the Conditional Generative Adversarial Network (C-GAN), and a hybrid C-VAEGAN that combines both. Each is trained on the Chinese-MNIST dataset (15 classes, \~1,000 images per class) and evaluated on a joint suite of generation quality metrics --- Fréchet Inception Distance (FID), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) --- as well as downstream classification accuracy on a Convolutional Neural Network (CNN) trained with the augmented data. A novel two-stage quality filter --- designed independently by the author --- is introduced to curate the synthetic samples before augmentation. The paper presents the full mathematical foundations of each architecture, explains all evaluation metrics in depth, and lays out a controlled experimental design capable of answering the central question: which generative paradigm best serves the augmentation use case?

# 1. Introduction

## 1.1 The Problem: Data Scarcity in a High-Cardinality Domain

The Chinese writing system is, by any measure, one of the most complex visual languages humans have ever produced. The Unicode CJK (Chinese, Japanese, Korean) block alone contains over 20,000 characters, each a unique configuration of strokes layered and balanced according to compositional rules developed over four millennia. For a machine learning model --- specifically a Convolutional Neural Network (CNN), a deep learning architecture that learns to recognize patterns by scanning images through layers of learnable filters --- recognizing all these characters reliably requires a large number of labeled training examples per class.

This is the core problem. Collecting and labeling thousands of handwritten examples per character is expensive and slow. When training data is sparse, CNNs tend to *overfit*: rather than learning the underlying geometric logic of a character --- its stroke structure, its proportions, its radical components --- the model memorizes the specific pixel patterns it has seen and fails to generalize to new examples. The result is a classifier that performs well on its training set but poorly in the real world.

One powerful remedy is **data augmentation**: artificially expanding the training set by creating new examples. Simple augmentation applies geometric transforms to existing images --- rotating, flipping, cropping, adding noise. These are useful, but they are bounded by the original data: every augmented image is still a transformation of something that already existed. **Generative data augmentation** is more ambitious. Instead of transforming existing images, we train a generative model --- a neural network that learns the underlying distribution of the data --- and sample entirely new examples from that learned distribution. Done well, the synthetic images are not copies or distortions of training examples; they are novel, plausible instances that the model has, in a meaningful sense, imagined.

## 1.2 Three Architectures, One Question

Three families of conditional generative models have emerged as strong candidates for this task, each with a distinct philosophy:

**The Conditional Variational Autoencoder (C-VAE) \[1, 2\]** takes a probabilistic approach. It learns to compress images into a structured, low-dimensional *latent space* --- a kind of internal coordinate system where similar characters cluster together --- and then reconstructs images from points in that space. The "conditional" part means the model is always told which character class it is working with, so generation is class-specific. The VAE is mathematically principled and trains stably, but its outputs can be slightly blurry because it optimizes a pixel-level reconstruction objective.

**The Conditional Generative Adversarial Network (C-GAN) \[3, 4\]** takes an adversarial approach. Two networks --- a Generator that creates fake images and a Discriminator that tries to tell fakes from real ones --- compete in a minimax game. Over time, the Generator learns to produce images realistic enough to fool the Discriminator. GANs tend to produce sharper, more photorealistic outputs than VAEs, but they are notoriously harder to train: the adversarial balance can tip, leading to *mode collapse* (where the Generator finds a few images that fool the Discriminator and stops exploring) or outright training instability.

**The Conditional VAEGAN (C-VAEGAN) \[5, 6\]** is a hybrid that aims to capture the best of both. It uses the VAE's encoder-decoder structure to maintain a coherent latent space, but replaces the VAE's blurry pixel-level reconstruction loss with the GAN's perceptual "does this look real?" criterion. Theoretically, this should yield outputs that are both structurally coherent *and* visually sharp. Whether this advantage holds empirically, on Chinese character data specifically, is one of the central questions this paper investigates.

This paper implements all three architectures on a shared dataset, evaluates them on a shared metric suite, and compares their downstream utility for data augmentation. The central question is empirical and deliberately open: *which generative paradigm best serves the augmentation use case for Chinese character recognition?*

## 1.3 Research Hypotheses

**H₁ (Generation Quality):** The C-VAEGAN will achieve a lower Fréchet Inception Distance (FID) and higher Structural Similarity (SSIM) and perceptual similarity (LPIPS) scores than either the C-VAE or C-GAN individually.

**H₂ (Augmentation Efficacy):** A CNN classifier trained on real data augmented with C-VAEGAN synthetic samples will achieve significantly higher validation accuracy than one trained on real data alone, with p \< 0.05 on a paired t-test across five independent experimental runs.

**H₃ (Quality Filter Value):** The novel two-stage quality filter introduced in this paper will improve augmentation efficacy relative to using unfiltered synthetic samples, for all three generative architectures.

# 2. Related Work

## 2.1 Variational Autoencoders

The Variational Autoencoder (VAE), introduced by Kingma and Welling in 2014 \[1\], was a landmark contribution to generative modeling. At its core, a VAE is an encoder-decoder architecture: the encoder maps an input image to a probability distribution in a low-dimensional latent space, and the decoder reconstructs the image from a sample drawn from that distribution. The key insight is that by forcing the latent distribution to remain close to a standard normal distribution, the model learns a smooth, continuous latent space in which nearby points decode to visually similar images --- a structure that makes controlled, interpolated generation possible.

Sohn, Lee, and Yan \[2\] extended this framework with conditioning, producing the Conditional VAE (C-VAE). By feeding the class label into both the encoder and decoder, the C-VAE can generate examples of a specific class on demand --- exactly what is needed for a labeled augmentation pipeline.

## 2.2 Generative Adversarial Networks

Goodfellow and colleagues introduced Generative Adversarial Networks (GANs) in 2014 \[3\] with a deceptively simple idea: train two networks simultaneously. A Generator network G learns to map random noise vectors to realistic-looking images; a Discriminator network D learns to distinguish real images from G's fakes. Each improves by trying to beat the other. When training converges, G has learned to produce images indistinguishable from the real data distribution. Mirza and Osindero \[4\] introduced the conditional extension (C-GAN) by providing the class label as an additional input to both G and D, enabling class-specific generation. Gulrajani and colleagues \[7\] later introduced the Gradient Penalty technique (WGAN-GP), which dramatically stabilizes GAN training by penalizing the Discriminator for violating a smoothness constraint --- addressing one of the most common failure modes in adversarial training.

## 2.3 VAE-GAN Hybrids

Larsen and colleagues \[5\] proposed combining the two paradigms: use the VAE's encoder-decoder as the generative backbone, but replace the VAE's pixel-level reconstruction loss with a learned perceptual loss computed from the GAN Discriminator's internal feature representations. The intuition is that pixel-level loss penalizes the model for every slightly misplaced pixel --- leading to blurriness as a conservative hedge --- whereas a perceptual loss asks only whether the image looks right overall. The conditional variant (C-VAEGAN) is directly applicable to Chinese character synthesis, where we need both structural regularity (favoring the VAE's latent structure) and visual sharpness (favoring the GAN's perceptual criterion).

## 2.4 Chinese Character Recognition and Synthesis

Deep CNN-based recognition of Chinese handwriting has advanced substantially, with state-of-the-art systems achieving over 95% accuracy on benchmark datasets \[8\]. However, these results depend on very large labeled datasets --- the CASIA-HWDB (Center for Analysis and Statistics of Handwriting -- Handwritten Database) corpus, for instance, contains nearly 3.9 million character images across 7,356 classes \[9\]. The Chinese-MNIST dataset \[10\], a more compact benchmark with 15 character classes and approximately 1,000 images each, provides a tractable starting point for generative augmentation experiments.

Kong and Xu \[6\] directly addressed Chinese character synthesis using a C-VAEGAN architecture, demonstrating feasibility on a small 200-samples-per-class regime. Their work encountered training instability --- discriminator loss collapsing to near zero, and KL (Kullback--Leibler) divergence numerical instability after approximately eight training epochs --- and reported no downstream augmentation evaluation. This paper extends their work in three directions: a systematic three-model comparison, a downstream classification evaluation, and the introduction of a quality filtering pipeline.

## 2.5 Positioning of This Work

To the author's knowledge, no prior published work has conducted a controlled three-way comparison of C-VAE, C-GAN, and C-VAEGAN specifically for Chinese character data augmentation with downstream OCR (Optical Character Recognition) accuracy as the evaluation criterion. This comparison directly addresses a gap in the literature: not which model generates the prettiest images, but which generative paradigm most usefully serves a downstream classification task.

# 3. Research Problem Statement

We begin with a labeled training set Dₛₑₐℓ = {(xᵢ, yᵢ)} of Chinese character images, where each xᵢ is a grayscale image and each yᵢ is one of C character class labels. Our generative model --- whichever of the three architectures we are testing --- is trained on this real data and used to produce a synthetic dataset Dₛᵧₙₜℏ. We then train a CNN classifier on the augmented set Dₛₑₐℓ ∪ Dₛᵧₙₜℏ and compare its performance to a baseline CNN trained on Dₛₑₐℓ alone. The central question is:

> *Which conditional generative architecture --- C-VAE, C-GAN, or C-VAEGAN --- produces synthetic data that most effectively augments the real training set for downstream Chinese character classification?*

This decomposes into three measurable sub-questions: (1) Which architecture produces the highest-quality synthetic characters, as measured by FID, SSIM, and LPIPS? (2) Which produces the greatest lift in CNN classifier accuracy when added to the real training data? (3) Does the novel two-stage quality filter improve augmentation efficacy, and does its benefit differ across architectures?

**Primary dataset:** Chinese-MNIST --- 15 character classes representing the Chinese words for the numbers zero through 100 million (*零、一、二、三、四、五、六、七、八、九、十、百、千、万、亿*), 1,000 grayscale 64×64-pixel images per class, 15,000 images total.

**Performance criteria:** Generation quality (FID, SSIM, LPIPS), downstream CNN classification accuracy, and paired t-test statistical significance (α = 0.05) across five independent runs.

# 4. Evaluation Metrics

Before describing the architectures, it is worth pausing to explain precisely how we will measure success. A generative model can fail in many ways: its images might look nothing like real characters, they might all look like the same character regardless of the conditioning label, or they might be technically realistic but somehow not useful for training a classifier. Each of our three metrics captures a different dimension of this question.

## 4.1 Fréchet Inception Distance (FID)

The Fréchet Inception Distance, or FID, asks: how similar is the *distribution* of generated images to the distribution of real images? It is the most widely used metric for evaluating generative models, and for good reason --- a model that produces high-quality individual images but misses whole regions of the real distribution (mode collapse) will score poorly on FID even if individual samples look good.

To compute FID, both sets of images --- real and generated --- are passed through a pretrained Inception-v3 network (a deep CNN trained on ImageNet), and the activations from one of its intermediate layers are extracted. These activations form a high-dimensional feature space in which images with similar visual characteristics cluster together. We model each set's feature activations as a multivariate Gaussian distribution, characterized by a mean vector μ and covariance matrix Σ. The FID is then the Fréchet distance between these two Gaussians:

*FID = \|\|μᵣ -- μᵔ\|\|\^2 + Tr(Σᵣ + Σᵔ -- 2(ΣᵣΣᵔ)\^(1/2))*

Here, μᵣ and Σᵣ are the mean and covariance of the real image features, and μᵔ and Σᵔ are those of the generated image features. Tr(·) is the matrix trace. The term \|\|μᵣ -- μᵔ\|\|\^2 measures how far apart the centers of the two distributions are; the trace term measures how well their shapes match. A lower FID is better --- a perfect generative model that exactly reproduces the real distribution would score zero. In practice, FID scores below 10 are considered excellent; scores above 50 suggest significant quality problems.

## 4.2 Structural Similarity Index Measure (SSIM)

While FID measures distributional similarity at the population level, the Structural Similarity Index Measure (SSIM) measures the similarity between two individual images. Introduced by Wang and colleagues \[11\], SSIM is motivated by a simple observation: the human visual system is particularly sensitive to structural information --- the spatial arrangement of luminance patterns --- rather than absolute pixel values. A metric that penalizes every slightly misaligned pixel equally (like mean squared error) misses this point.

SSIM decomposes image similarity into three components --- luminance (l), contrast (c), and structure (s) --- and combines them multiplicatively:

*SSIM(x, y) = l(x,y) · c(x,y) · s(x,y)*

where each component is computed over local image patches. The luminance term l(x,y) = (2μₓμᵧ + C₁) / (μₓ² + μᵧ² + C₁) compares mean pixel values; the contrast term c(x,y) = (2σₓσᵧ + C₂) / (σₓ² + σᵧ² + C₂) compares standard deviations; and the structure term s(x,y) = (σₓᵧ + C₃) / (σₓσᵧ + C₃) compares the normalized cross-correlation between patches. C₁, C₂, C₃ are small stabilization constants. The result is a value in \[-1, 1\] where 1 means identical images, 0 means no structural correlation, and negative values indicate structural anti-correlation. For Chinese character generation, an SSIM above \~0.6 between a generated image and its nearest real-class neighbor indicates the generated character preserves the essential stroke structure.

## 4.3 Learned Perceptual Image Patch Similarity (LPIPS)

SSIM is a hand-crafted metric; Learned Perceptual Image Patch Similarity (LPIPS), introduced by Zhang and colleagues \[12\], asks instead: what does a deep neural network think about the similarity between two images? The idea is that a network trained on large amounts of image data has learned feature representations that correlate with human perceptual judgment far better than any hand-designed formula.

To compute LPIPS, both images are passed through a pretrained deep network (such as VGG or AlexNet). The activations at multiple layers are extracted, normalized, and compared channel-by-channel. The differences are then aggregated into a single scalar distance:

*LPIPS(x, y) = Σₗ wₗ \|\|φₗ(x) -- φₗ(y)\|\|\^2*

where φₗ(·) denotes the feature activations at layer l, and wₗ are learned weights that calibrate the contribution of each layer. Unlike SSIM, LPIPS is a *distance* --- lower is better, meaning the two images are perceptually more similar. A generated Chinese character with a low LPIPS distance to a real character of the same class not only has the right pixel structure, it looks right to a network that has learned something like visual perception.

## 4.4 Downstream CNN Classification Accuracy

All three of the above metrics evaluate the quality of generated images in isolation. But the ultimate test for our purposes is pragmatic: does adding the synthetic data actually help a classifier learn? We measure this by training a CNN classifier under two conditions --- on real data alone (the baseline) and on real plus synthetic data (the augmented condition) --- and comparing their validation accuracy. To ensure the comparison is statistically reliable, we repeat each training run five times with different random seeds and apply a paired t-test to determine whether any observed improvement is significant (p \< 0.05) or within the noise of training variability. We also compute Cohen's d, a measure of effect size that tells us not just whether the difference is statistically significant, but how large it is in practical terms.

# 5. Method

## 5.1 Experimental Pipeline Overview

All three architectures share the same experimental scaffold, which is essential for a fair comparison. The pipeline has four stages: (1) train the generative model on the real training set; (2) generate a synthetic dataset; (3) optionally filter the synthetic dataset through the quality pipeline; (4) train the CNN classifier on the combined real-plus-synthetic set and evaluate on the held-out validation set. By holding everything constant except the generative architecture, any observed difference in downstream accuracy is attributable to that architecture and not to any other experimental variable.

  -------------------- ----------------------------------------- ------------------------------------- ------------------------------------------------
  **Property**         **C-VAE**                                 **C-GAN**                             **C-VAEGAN (Hybrid)**

  Training objective   ELBO (reconstruction + β·KL divergence)   Minimax adversarial game              ELBO + adversarial perceptual loss

  Latent space         Structured, continuous N(0,I) prior       Unstructured noise vector z           Structured (VAE encoder) + GAN perceptual loss

  Output quality       Smooth; may be slightly blurry            Sharp; risk of mode collapse          Sharp and structured (theoretical best)

  Training stability   High --- ELBO is well-defined             Medium --- collapse risk without GP   Medium-High --- GP regularization required

  Key citations        \[1\], \[2\]                              \[3\], \[4\], \[7\]                   \[5\], \[6\]
  -------------------- ----------------------------------------- ------------------------------------- ------------------------------------------------

## 5.2 Mathematical Foundations of Each Architecture

**The C-VAE and the Evidence Lower Bound (ELBO)**

The C-VAE rests on a beautiful probabilistic idea: rather than learning a fixed encoding for each image, learn a *distribution* over possible encodings. Formally, given an image x and its class label y, the encoder produces the parameters (μ, σ) of a Gaussian distribution qφ(z \| x, y) over a latent vector z. The decoder then takes a sample z from this distribution (along with y) and reconstructs the image. The model is trained to maximize the Evidence Lower Bound (ELBO), which balances two competing objectives:

*ELBO = E\_{qφ(z\|x,y)} \[log pθ(x \| z, y)\] − β · KL(qφ(z \| x, y) \|\| p(z))*

The first term is the **reconstruction loss**: how well does the decoder reconstruct the original image from the sampled z? The second term is the **KL (Kullback--Leibler) divergence**: how far is the learned encoding distribution qφ(z \| x, y) from a standard normal distribution p(z) = N(0, I)? The KL term acts as a regularizer that keeps the latent space organized --- preventing the model from using arbitrary, scattered encodings. The hyperparameter β controls the trade-off: higher β produces a smoother, more regular latent space at some cost to reconstruction sharpness.

**The Reparameterization Trick**

There is a subtle but critical engineering challenge lurking in the ELBO: to optimize the model with gradient descent, we need to backpropagate gradients through the *sampling step* z \~ qφ(z \| x, y). But sampling is a stochastic operation, and stochastic operations are not differentiable in the usual sense. Gradients cannot flow through randomness.

The **reparameterization trick** is the elegant solution. Instead of sampling z directly from the learned distribution N(μ, σ²), we rewrite the sampling operation as:

*z = μ + σ ⊙ ε, ε \~ N(0, I)*

Here, ε is a sample from a *fixed* standard normal distribution --- it carries all the randomness --- while μ and σ are deterministic outputs of the encoder network. Think of it this way: imagine you need to draw a random point from a circle of radius r centered at point p. You could sample directly, but gradients would not flow. Instead, you sample a random direction ε from a unit circle, then *compute* the point as p + r·ε. The randomness lives in ε, which we treat as a fixed input, not a parameter; all the learnable structure lives in p and r, through which gradients flow freely. The reparameterization trick shifts the randomness out of the computational graph and into a fixed noise term, making the entire operation differentiable with respect to μ and σ.

**The C-GAN and Gradient Penalty**

The C-GAN trains a Generator G and Discriminator D simultaneously with opposing objectives. Conditioned on class label y, G maps a noise vector z to a synthetic image G(z \| y); D takes an image (real or fake) and its label y and outputs a probability that the image is real. The minimax objective is:

*min_G max_D \[ E\[log D(x \| y)\] + E\[log(1 − D(G(z \| y)))\] \]*

The Generator tries to minimize this (fool D); the Discriminator tries to maximize it (detect fakes). To stabilize training, we apply the Wasserstein Gradient Penalty (WGAN-GP) \[7\], which adds a term penalizing the Discriminator for having gradients too far from unit norm:

*Lᵄ += λ · E\[(\|\|∇ᵄ D(x̂ \| y)\|\|\_2 − 1)\^2\]*

where x̂ is a random convex combination of real and generated images. This constraint keeps the Discriminator smooth, preventing the gradient signals sent to the Generator from exploding or vanishing --- the root cause of most GAN training instability.

**The C-VAEGAN Combined Objective**

The C-VAEGAN uses the VAE encoder-decoder as its generative backbone, but replaces the VAE's pixel-level reconstruction loss with a perceptual loss computed from the GAN Discriminator's internal feature representations. The combined loss is:

*L = Lᴼᴹᴸᴺ + λₐᵈᵛ · Lᴺᴬᴺ + λᶠᵉᵃᵗ · Lᶠᵉᵃᵗᵣᵐ*

The ELBO term maintains the structured latent space; the adversarial term sharpens outputs by asking the Discriminator whether they look real; the feature-matching term aligns the internal representations of real and reconstructed images within the Discriminator's layers, further improving perceptual quality.

## 5.3 Shared Architecture

To ensure a fair comparison, all three models share the same convolutional backbone. The encoder (used in C-VAE and C-VAEGAN) consists of four convolutional layers with Batch Normalization and LeakyReLU activations, reducing the input from 64×64 to a 4×4 feature map, then projecting to the latent parameters (μ, log σ²). The class label is concatenated to the image as a tiled channel before the first convolutional layer. The decoder/generator uses four transposed convolutional layers (the "reverse" of convolution, used to upscale feature maps back to image size), producing a 64×64 grayscale output through a Tanh activation that maps values to \[-1, 1\]. The discriminator uses the same four-layer convolutional structure as the encoder, ending in a scalar logit.

## 5.4 The Two-Stage Quality Filter (Original Contribution)

A critical and independently developed contribution of this paper is the two-stage quality filter, designed to curate synthetic samples before they enter the augmentation pool. The motivation is straightforward: generative models --- particularly early in training or in underrepresented character classes --- do not always produce good samples. Adding low-quality synthetic images to the training set could harm the classifier rather than help it. Filtering before augmentation ensures that only synthetic samples meeting minimum quality standards are used.

To the author's knowledge, applying a two-stage discriminator-confidence-plus-perceptual-distance filter as a principled quality gate for generative data augmentation in the Chinese character domain has not been previously reported in the literature. The design was developed independently in the course of this project.

**Stage 1 --- Discriminator confidence:** A synthetic image x̃ is accepted only if the trained Discriminator assigns it a confidence score above a threshold τ_D (approximately the top 70th percentile of generated samples). This stage filters out samples that the model itself "knows" are poor --- images that look so far from the real distribution that even the Discriminator is not fooled.

**Stage 2 --- Perceptual proximity:** Among samples passing Stage 1, we additionally require that the SSIM score between x̃ and its nearest real-class neighbor exceeds τ_S, and that the LPIPS distance is below τ_L. This stage filters out samples that may score well on the Discriminator but are perceptually dissimilar from any real example in the class --- stylistically plausible but geometrically wrong. The thresholds are tuned on a held-out validation split.

The filter is applied identically across all three architectures, enabling a clean A/B comparison between filtered and unfiltered augmentation --- this is the basis for testing H₃.

## 5.5 CNN Classifier

The evaluation classifier is a standard three-block CNN: each block consists of two Conv2d layers with Batch Normalization and ReLU activations followed by MaxPooling, reducing spatial dimensions from 64×64 down to an 8×8 feature map. Two fully connected layers with Dropout (p = 0.4) and a softmax output over 15 classes complete the architecture. The same classifier is used in all conditions --- baseline and all three augmented variants --- trained with the Adam optimizer and a cosine learning rate schedule. Holding the classifier fixed ensures that differences in accuracy between conditions reflect the quality of the augmentation, not the classifier design.

## 5.6 Hyperparameter Search

  -------------------- ------------------- ------------------------------------------------------------
  **Parameter**        **Candidates**      **Selection Criterion / Applies To**

  Latent dimension d   32, 64, 128         FID on held-out set --- C-VAE, C-VAEGAN

  β (KL weight)        0.5, 1.0, 2.0       Reconstruction vs. diversity trade-off --- C-VAE, C-VAEGAN

  λₐᵈᵛ (GAN weight)    0.1, 0.5, 1.0       Output sharpness vs. training stability --- C-VAEGAN only

  GP penalty λ         1, 5, 10            Training stability --- C-GAN, C-VAEGAN

  Augmentation ratio   0.25×, 0.5×, 1.0×   CNN validation accuracy --- all three architectures
  -------------------- ------------------- ------------------------------------------------------------

# 6. Experimental Results

*NOTE: Placeholders per midpoint check requirements. Full results for final submission.*

## 6.1 Generation Quality Comparison

*\[PLACEHOLDER: Table: FID scores for C-VAE, C-GAN, C-VAEGAN across latent dimension candidates --- which architecture scores lowest?\]*

*\[PLACEHOLDER: Table: Mean SSIM and LPIPS per architecture per character class\]*

*\[PLACEHOLDER: Visual grid: 5 generated samples per class × 3 architectures, side-by-side\]*

## 6.2 Training Dynamics

*\[PLACEHOLDER: Loss curves: ELBO components (C-VAE/C-VAEGAN), D/G loss (C-GAN/C-VAEGAN). Does gradient penalty stabilize training?\]*

*\[PLACEHOLDER: Mode collapse diagnostic for C-GAN: per-class Jensen-Shannon divergence between generated and real distributions\]*

## 6.3 Quality Filter Analysis

*\[PLACEHOLDER: Filter pass rates per architecture per class (Stage 1 and Stage 2 separately)\]*

*\[PLACEHOLDER: Distribution of SSIM scores: filtered vs. unfiltered per model\]*

## 6.4 Downstream CNN Accuracy

*\[PLACEHOLDER: Table: Mean ± std validation accuracy across 5 seeds --- Baseline \| +C-VAE (filtered) \| +C-GAN (filtered) \| +C-VAEGAN (filtered)\]*

*\[PLACEHOLDER: Paired t-test results: p-value and Cohen's d for H₁, H₂, H₃\]*

*\[PLACEHOLDER: Ablation: filtered vs. unfiltered augmentation per architecture (test of H₃)\]*

*\[PLACEHOLDER: Accuracy vs. augmentation ratio (0.25×, 0.5×, 1.0×) per architecture\]*

*\[PLACEHOLDER: Confusion matrices: baseline condition and best-performing augmented condition\]*

# 7. Conclusions

*\[PLACEHOLDER: To be completed for final submission. Will address: (1) Which architecture won on generation quality and augmentation efficacy? (2) Were H₁/H₂/H₃ supported? (3) What does the quality filter analysis reveal? (4) Limitations and future directions, including potential extension to a larger CASIA-HWDB subset.\]*

# 8. Request for Cloud GPU Compute

This section provides a brief justification and estimate for GPU time requested in support of this project.

## 8.1 Why GPU Compute is Necessary

All three generative models in this study --- C-VAE, C-GAN, and C-VAEGAN --- are deep convolutional neural networks trained on image data. Their training requires large numbers of matrix multiplications over high-dimensional tensors, a task for which a Graphics Processing Unit (GPU) is orders of magnitude faster than a Central Processing Unit (CPU). A training run that takes 2--3 hours on a modern GPU would take days on CPU, making iterative development and hyperparameter search practically impossible without GPU access.

## 8.2 Workload Breakdown

The project involves three distinct computational workloads:

**Generative model training (primary cost).** Three models × three hyperparameter candidates each (latent dimension, β, λ) = approximately 9--15 training runs per model family. Each run on Chinese-MNIST (15 classes, 15,000 images) at 50--100 epochs with batch size 128 is estimated at 1--3 hours per run on a mid-range GPU (e.g., NVIDIA V100 or A100). Total estimated training time: 30--70 GPU-hours for the full hyperparameter search across all three architectures.

**CNN classifier training (secondary cost).** Five seeds × four conditions (baseline + three augmented) = 20 classifier training runs per augmentation ratio × three ratios = 60 runs total. Each CNN training run is approximately 15--30 minutes at 30 epochs. Total estimated time: 15--30 GPU-hours.

**Quality filter and metric computation.** FID requires generating \~1,000 images per class and running Inception-v3 inference; LPIPS requires multiple forward passes per image pair. Estimated at 2--5 GPU-hours total across all evaluations.

## 8.3 Summary Request

  ------------------------------------------------------ ------------------------- --------------------------------------------
  **Workload**                                           **Estimated GPU-Hours**   **Notes**

  Generative model training (all 3 models + HP search)   30--70 hours              Dominant cost; parallelizable across runs

  CNN classifier training (60 runs, 5 seeds)             15--30 hours              Short runs; moderate total

  Metric computation (FID, LPIPS, quality filter)        2--5 hours                Inference-only; relatively light

  Total (conservative estimate)                          \~50--105 hours           Request: \~120 GPU-hours to include buffer
  ------------------------------------------------------ ------------------------- --------------------------------------------

A request of **120 GPU-hours** is submitted. This provides sufficient capacity for the full hyperparameter search, the five-seed statistical evaluation, and a comfortable development buffer for debugging and iteration. If time allows, the buffer would also support an exploratory extension to a small CASIA-HWDB subset as a secondary experiment. All code will be written in PyTorch and is designed to run on any NVIDIA GPU with at least 16 GB VRAM; the dataset (Chinese-MNIST) is publicly available and small (\~100 MB).

# Acknowledgements

This paper emerged from several months of sustained, iterative dialogue with Claude (Anthropic), used here as a research thinking partner rather than a writing tool. The core research question, the choice of Chinese characters as a domain, the three-model comparative framing, and the experimental hypotheses all originated in my own thinking; Claude's role was to sharpen those ideas through rigorous back-and-forth, flag gaps in reasoning, suggest implementation strategies, and help translate rough intuitions into precise technical language. The architecture decisions, the quality filter pipeline, and the dataset selection rationale each went through multiple rounds of challenge and refinement in that dialogue. I find this mode of working --- bringing your own ideas and using AI to stress-test and develop them --- to be a genuinely productive form of intellectual partnership, and wanted to acknowledge it honestly rather than obscure it.

# References

\[1\] D. P. Kingma and M. Welling, \"Auto-Encoding Variational Bayes,\" in Proc. ICLR, 2014. arXiv:1312.6114.

\[2\] K. Sohn, H. Lee, and X. Yan, \"Learning Structured Output Representation using Deep Conditional Generative Models,\" in Proc. NeurIPS, vol. 28, 2015.

\[3\] I. J. Goodfellow et al., \"Generative Adversarial Nets,\" in Proc. NeurIPS, vol. 27, 2014. arXiv:1406.2661.

\[4\] M. Mirza and S. Osindero, \"Conditional Generative Adversarial Nets,\" arXiv:1411.1784, 2014.

\[5\] A. B. L. Larsen, S. K. Sønderby, H. Larochelle, and O. Winther, \"Autoencoding beyond Pixels using a Learned Similarity Metric,\" in Proc. ICML, 2016. arXiv:1512.09300.

\[6\] B. Kong and Y. Xu, \"Generative Adversarial Networks for Chinese Character Image Synthesis,\" unpublished manuscript, 2021.

\[7\] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, \"Improved Training of Wasserstein GANs,\" in Proc. NeurIPS, vol. 30, 2017. arXiv:1704.00028.

\[8\] C. Zhang, P. Yin et al., \"Chinese Character Recognition with Deep CNNs,\" in Proc. ICDAR, 2017.

\[9\] C.-L. Liu, F. Yin, D.-H. Wang, and Q.-F. Wang, \"CASIA Online and Offline Chinese Handwriting Databases,\" in Proc. ICDAR, 2011.

\[10\] Chinese-MNIST Dataset. Kaggle. \[Online\]. Available: https://www.kaggle.com/datasets/gpreda/chinese-mnist.

\[11\] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, \"Image Quality Assessment: From Error Visibility to Structural Similarity,\" IEEE Trans. Image Process., vol. 13, no. 4, pp. 600--612, 2004.

\[12\] R. Zhang et al., \"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,\" in Proc. CVPR, 2018. arXiv:1801.03924.



