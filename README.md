# Facial Expression Classification with Deep Learning

## Project Overview

This project aims to classify facial expressions into categories such as anger, happy, sad, etc., using various deep learning models. We experimented with several architectures, including VGG19, MobileNetV3, and ResNet50. By applying advanced techniques like Inception Pyramid Pooling Module (iPPM) to enhance feature extraction, we achieved the highest accuracy with the ResNet50 backbone.

## Models and Accuracy

The following models were evaluated:

| Model Name               | Accuracy |
|--------------------------|----------|
| VGG19                    | 0.4120   |
| MobileNetV3Small         | 0.4247   |
| ResNet50                 | 0.4729   |
| Advanced Model (using iPPM) | 0.5269   |

The highest accuracy was achieved with the ResNet50 model. Therefore, ResNet50 was used as the backbone for the advanced model, which incorporates iPPM to further refine feature extraction.

## Inception Pyramid Pooling Module (iPPM)

iPPM is used to enhance the features extracted by the VGG19 model. The iPPM process involves:

1. **Upsampling**: Nearest neighbor upsampling is applied to the input tensor with various upsampling factors.
2. **Max Pooling**: Max pooling is performed on the upsampled tensor with spatial dimensions matching the upsampling factor. 
3. **Convolution**: A convolutional layer with 128 filters is applied to the pooled tensor to produce refined feature maps.

The formula for the iPPM operation is as follows:
F_i = Conv(MaxPool(UpSample(I, s_i)), k_i)

where:
- `F_i` represents the feature map at level `i`,
- `s_i` is the upsampling factor,
- `k_i` is the convolutional kernel size,
- `I` is the input tensor.

## Dataset

The dataset used for training is the [Facial Expressions Training Data - Noam Segal](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data?select=anger), which contains approximately 28,000 images. Augmentation techniques such as flipping, rotating, shearing, and zooming were applied to generate a total of 50,000 images for modeling.

## Training Details

- **Batch Size**: 64
- **Epochs**: 50
- **Early Stopping Patience**: 5
- **Learning Rate**: 0.001

## Results

The Advanced Model, built using ResNet50 and iPPM, achieved a 5% increase in accuracy compared to the baseline ResNet50 model.

## Reference

The methodology and techniques applied in this project are inspired by the following reference:

- Haque, Rezuana & Hassan, Md.Mehedi & Bairagi, Anupam & Shariful Islam, Sheikh Mohammed. (2024). NeuroNet19: an explainable deep neural network model for the classification of brain tumors using magnetic resonance imaging data. *Scientific Reports*, 14. [10.1038/s41598-024-51867-1](https://doi.org/10.1038/s41598-024-51867-1).

## How to Use

1. **Install Dependencies**: Ensure you have all the necessary libraries installed (e.g., TensorFlow, OpenCV, pandas).
2. **Load the Dataset**: Download and preprocess the dataset as described above.
3. **Train the Model**: Use the provided scripts to train the models with the specified parameters.
4. **Evaluate the Model**: Test the models on validation or test data to assess performance.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Your feedback and improvements are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
