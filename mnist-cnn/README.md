# CNN for MNIST Handwritten Digits Classification

## Project Overview
This project demonstrates the development of *simple deep-learning models* for recognition of handwritten digits in the MNIST dataset using the classical computer vision algorithm - *The Convolutional Neural Networks (CNNs)*, achieving near to 100% accuracy on both train and test datasets.

### Key Achievements
- **Exceptional Performance**: 99.89% and 99.06% accuracy on train and test dataset respectively.
- **Experiment on Hyperparameters**: Gained insights about how to experiment with the hyperparameters for model tuning.

## Dataset Information
- **Source**: torchvision's `torchvision.datasets`
- **Size**: 60000 train images and 10000 test images

## Technical Approach

### 1. Download and Transform Dataset
- Dataset required for this project was downloaded from `torchvision.datasets.MNIST`
- Apply transformations:
    - conversion from PIL Image to Tensors
    - Normalizing the pixel values to mean and standard deviation of the dataset (mean=0.1307 std=0.3081)

### 2. Model Development
- **Model**: Built multiple CNN models by subclassing `nn.Module`
- **Loss Function**: Employed *Cross Entropy Loss function* as the empirical loss for the problem (`nn.CrossEntropyLoss`)
- **Optimizer**: Experimented with two model popular optimizers - the *Stochastic Gradient Descent* and *Adam* Optimizers (`torch.optim.SGD()` and `torch.optim.Adam()`)
- **Regularization**: Employed only *dropout* technique for simplicity.

### 3. Optimization
Experimented with the following hyperparameters:
- LEARNING RATE (default, 0.05, 0.1, 0.0008)
- EPOCHS (10, 20, 25, 30, 40)
- BATCH SIZE (32, 64, 512)

Then chose the best performing model with its corresponding hyperparameters.

## Model Performance
- All the models trained and saved achieve accuracy of more than 95% on both train and test sets.
- When experimented with the hyperparameters, unstable combinations lead to bad accuracies (sometimes 30-40%)

## Limitation and Challenges to Deployment
The MNIST dataset is a *highly preprocessed data* of images with *digits centered* and multiple transformations on the image to improve image quality. When models trained on such data are deployed in the wild, they face challenges like - poor quality, incompatible image sizes and digits pushed to corners or borders.

This poses another computer vision problem, digit recognition and image processing. Mastering or developing a solution for this problem would allow easy deployment of models here in the repository.

## Project Value & Learning Outcomes

### Technical Skills Demonstrated
- **Deep Learning Fundamentals**: Implemented CNNs from scratch using PyTorch, gaining hands-on experience with convolutional, pooling, activation, and fully connected layers.
- **Model Optimization**: Experimented systematically with learning rate, batch size, and epochs to understand their effect on convergence and generalization.
- **Training & Evaluation Workflow**: Designed modular training loops and evaluation processes, integrating loss monitoring, accuracy tracking, and checkpointing of models.
- **Software Engineering Practices**: Built reusable and well-structured codebases by subclassing nn.Module, utilizing optimizers, and saving models for reproducibility.

### Industry Relevance
- **Core Computer Vision Skillset**: CNN-based digit classification is a fundamental benchmark that translates into real-world applications such as OCR systems, document scanning, and financial transaction automation.

- **Hyperparameter Tuning Insights**: Understanding trade-offs when tuning deep learning hyperparameters mirrors industry ML workflows where model stability and efficiency are critical.

- **Model Deployment Awareness**: Identified limitations of MNIST in production environments (image noise, alignment, size variance), reflecting deployment challenges often faced by organizations working with messy, real-world datasets.

## Repository Structure


```
mnist-cnn/
├── checkpoints/
│   ├── best_model.pt 
│   ├── CNN1_1.pt
│   ├── CNN1_2.pt
│   ├── CNN2_1.pt
│   ├── CNN2_2.pt
│   ├── CNN3_(0.7_dropout).pt
│   ├── CNN3_(2x_bs_lr).pt
│   ├── CNN3_(Batchsize128_LR0.004_40EPOCHS).pt
│   ├── CNN3_1.pt
│   └── PRIMITIVE_CNN.pt
├── data/MNIST/raw
├── images-for-notebooks/
├── notebooks/
├── real-world-images/
├── scripts/
├── selected_model/
├── src/
├── training-plots/
└── README.md                                               # This documentation
```

## Professional Portfolio Value
- **Portfolio Demonstration**: This project serves as a demonstrative portfolio piece showcasing technical competence in PyTorch, CNNs, and systematic experimentation.
- **Interview Utility**: Provides tangible talking points for technical interviews—such as explaining optimization choices, discussing failure cases, and strategies for improvement.
- **Versatility Showcase**: Reflects adaptability and potential for scaling from academic datasets (MNIST) to enterprise-scale use cases (OCR, healthcare imaging, computer vision in retail and security).

## Future Enhancements
- **Extended Dataset**: Apply the model to more challenging datasets like EMNIST (letters + digits) or SVHN (street numbers) to evaluate generality.
- **Data Augmentation**: Introduce random rotations, shifts, or noise injection to build robustness against real-world variations in handwriting and image quality.
- **Deployment Pipeline**: Create an end-to-end pipeline by integrating preprocessing, model inference, and user input through a Flask/Django web app or mobile application.

## Conclusion
This project successfully demonstrates the implementation of CNNs for digit recognition using the MNIST dataset, achieving near-perfect accuracy and strong generalization. Beyond achieving high accuracy, it served as a practical exercise in neural network construction, optimization, and evaluation, while offering insights into the challenges of deploying models in real-world environments.

**Key Takeaway**: Even though training the models to achieve wonderful accuracies is easy, deployment remains a major challenge.
---

*Project completed: Auguts 2025*  
*Technologies: Python, PyTorch, pandas, numpy, matplotlib*  
*Dataset: MNIST (60,000 train samples, 10,000 test samples, sourced via torchvision.datasets.MNIST)*