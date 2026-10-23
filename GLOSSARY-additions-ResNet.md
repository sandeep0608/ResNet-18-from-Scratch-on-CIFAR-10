# Glossary — ResNet / CIFAR-10 additions

New terms from the ResNet-18 on CIFAR-10 project. Add these to your existing `GLOSSARY.md` (they build on the LeNet/CNN terms already there). Grouped by topic to match the original.

---

## Dataset & data preparation

**CIFAR-10** — a dataset of 60,000 small (32×32) color images across 10 real-world classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Harder than MNIST because images are in color, show real objects, and vary a lot within each class.

**Data augmentation** — randomly altering training images each time they're shown (e.g. flipping or shifting) so the model sees more variety and learns general patterns instead of memorizing exact images. Reduces overfitting. Applied to the **training set only** — the test set stays clean.

**Random crop (with padding)** — pads the image with a small border, then takes a random 32×32 window from it. Nudges the object's position around so the model learns position shouldn't matter.

**Random horizontal flip** — randomly mirrors an image left-to-right during training. Valid because a flipped dog is still a dog (unlike a vertical flip, which would create unrealistic upside-down images).

**Regularization** — any technique that discourages a model from overfitting so it generalizes better to new data. Data augmentation and weight decay are both forms of regularization.

---

## Architecture

**Residual block** — the building block of ResNet: two `conv → BatchNorm` layers (the "edits", F(x)) plus a **skip connection** that adds the block's input back to its output. Output = ReLU(F(x) + x).

**Residual (skip) connection / shortcut** — the path that carries a block's input *around* its layers and adds it to the output. Lets the block learn only the *change* to the input rather than a full transformation. The key innovation of ResNet.

**Identity mapping** — the case where a block passes its input through unchanged (the "edits" F(x) are zero). Skip connections make this easy to learn, which is why adding more layers never hurts a ResNet.

**Degradation problem** — the pre-ResNet observation that making a plain network deeper eventually made it *worse*, even on training data, because extra layers couldn't even learn to do nothing. Residual connections solved this.

**Vanishing gradients** — in very deep networks, the learning signal (gradient) from backpropagation fades to almost nothing by the time it reaches the early layers, so they stop learning. Skip connections give the gradient a direct path backward, which helps prevent this.

**Batch Normalization (BatchNorm / BN)** — a layer, used after each convolution, that re-centers and re-scales the values flowing through the network to keep them in a stable range. Makes deep networks train faster and more reliably. Usual pattern: conv → BN → ReLU.

**Stem** — the first convolution(s) at the start of the network, before the residual blocks begin. Here it expands the 3-channel color image up to 64 channels.

**Global average pooling** — collapses each final feature map down to a single number by averaging it, before the classifier. A parameter-efficient alternative to flattening the whole grid into a large fully-connected layer.

---

## Training & optimization

**SGD (Stochastic Gradient Descent)** — a classic optimizer that updates weights using the gradient of each batch. With momentum and weight decay, it often reaches higher final accuracy than Adam on image tasks like CIFAR.

**Momentum** — an add-on to SGD that carries some "velocity" from previous steps, like a ball rolling downhill. Helps the optimizer push through small bumps and settle into better solutions.

**Weight decay** — a small penalty that discourages weights from growing too large. A form of regularization that reduces overfitting.

**Learning-rate schedule** — a plan for changing the learning rate during training, typically starting large (fast progress) and shrinking it over time (fine settling). Often improves final accuracy.

**Cosine annealing** — a specific learning-rate schedule that smoothly lowers the learning rate from its start value toward zero following a cosine curve over the course of training.

**Checkpoint** — a saved copy of the model's weights, often saved whenever the model reaches a new best accuracy during training. Protects against losing progress and lets you keep the best version rather than just the final one.

---

*These extend the LeNet/CNN glossary. Terms already defined there (convolution, pooling, weights, loss, gradient descent, optimizer, learning rate, epoch, overfitting, stride, etc.) still apply and aren't repeated.*
