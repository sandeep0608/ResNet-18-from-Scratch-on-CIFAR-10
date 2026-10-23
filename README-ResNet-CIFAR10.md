# ResNet-18 from Scratch on CIFAR-10 (PyTorch)

A from-scratch PyTorch implementation of **ResNet**, the architecture introduced in He, Zhang, Ren & Sun (2015), *"Deep Residual Learning for Image Recognition."* Built as a ResNet-18 and trained to classify the 10 object classes of CIFAR-10.

> ResNet introduced the **residual (skip) connection** — the idea that let neural networks go from ~20 layers to 150+, and the foundation of most modern computer-vision models. This project implements the residual block by hand and stacks it into a working ResNet-18.

## Result

**85.75% test accuracy** on CIFAR-10 (10,000 unseen test images), trained for 15 epochs with the Adam optimizer.

*Improving further:* the training recipe here was kept simple. Switching to SGD with momentum + weight decay, adding a cosine learning-rate schedule, and training for ~30 epochs typically pushes this same architecture into the low 90s. (See "Possible improvements" below.)

## Dataset

CIFAR-10: 60,000 color images (32×32, 3 channels) across 10 classes — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. 50,000 for training, 10,000 for testing.

## Architecture (ResNet-18)

| Stage | Detail | Output |
|---|---|---|
| Stem | 3×3 conv, 3→64, + BatchNorm + ReLU | 64 × 32 × 32 |
| Layer 1 | 2 residual blocks, 64 channels | 64 × 32 × 32 |
| Layer 2 | 2 residual blocks, 128 channels, first one downsamples | 128 × 16 × 16 |
| Layer 3 | 2 residual blocks, 256 channels, first one downsamples | 256 × 8 × 8 |
| Layer 4 | 2 residual blocks, 512 channels, first one downsamples | 512 × 4 × 4 |
| Head | global average pool → fully-connected → 10 classes | 10 |

~11.2 million parameters.

**The residual block** (the core building block): two `conv → BatchNorm` layers (the "edits", F(x)) with a **skip connection** that adds the block's input back to its output — `output = ReLU(F(x) + x)`. When a block changes shape (channel count or downsampling), a 1×1 convolution on the shortcut matches dimensions so the addition works.

## How it works

- **Residual/skip connections** let each block learn only the *change* to its input and add the original back. This means a block can easily learn to "do nothing" (pass the input through), so adding depth never hurts — solving the degradation problem that made very deep plain networks train poorly. The shortcut also gives gradients a direct path backward, keeping deep networks trainable.
- **Batch normalization** after each convolution keeps the values flowing through the network in a stable range, which makes a deep network train faster and more reliably.
- **Data augmentation** (random crop + horizontal flip, applied to training images only) shows the model varied versions of each image every epoch, reducing overfitting and improving generalization.
- **Global average pooling** collapses each final feature map to a single number before the classifier, keeping the fully-connected head small (far fewer parameters than flattening the whole grid).

## Techniques used

- Residual (skip) connections
- Batch normalization
- Data augmentation (random crop, horizontal flip)
- Input normalization
- Global average pooling
- Model checkpointing (saving the best-performing weights during training)

## Tech stack

- Python
- PyTorch / torchvision
- Trained on a single GPU (Google Colab)

## How to run

1. Open the notebook in Google Colab (or Jupyter).
2. Enable a GPU runtime (Colab: *Runtime → Change runtime type → GPU*).
3. Run the cells top to bottom. CIFAR-10 downloads automatically on first run.

The best model's weights are saved as `resnet_cifar_best.pth`.

## Possible improvements

- **SGD + momentum + weight decay** instead of Adam (typically higher final accuracy on CIFAR).
- **Cosine learning-rate schedule** — start high, decay toward zero across training.
- **More epochs** (~30–50) to let the schedule play out.
- Stronger augmentation (e.g. random erasing, color jitter).

## What I learned

- The residual (skip) connection: why very deep "plain" networks degrade, and how adding the input back lets networks go much deeper.
- Batch normalization and why it stabilizes deep-network training.
- Data augmentation as a practical defense against overfitting — and why it's applied only to the training set.
- Global average pooling as a parameter-efficient alternative to large fully-connected heads.
- That improving a model is often about *how* you train it (optimizer, learning-rate schedule, epochs), not just the architecture.

## Reference

K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016. (arXiv:1512.03385)
