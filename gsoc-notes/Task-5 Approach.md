
## Problem Overview

Task-5 deals with 2 angles, one is classification and the other is anomaly detection. In this task I chose the classification approach, where my objective is to classify a set of simulated strong gravitational lensing images:

* With Sub-structure
* Without Sub-structure

The **Evaluation metric** for this task is the AUCROC (Area under the Receiver Operating Characteristic curve).

## How to Solve the problem

Well looking at the above problem description it is clear that we need to build a 
binary-classification algorithm, but before that we need to explore the data first. 

### Exploring the Target Distribution
![[Screenshot from 2023-04-03 02-21-14.png]]

Looking at the above plot it is clear that the data has each targets **equally distributed**. Still, it is a good idea to do stratification when splitting the data.

### Exploring the Image Data
![[Screenshot from 2023-04-03 02-24-00.png]]
Here the images are of size $(150, 150)$. The images are still gray-scale images but has a **max-pixel** value of $255$.

Now moving onto to the next step of the approach, i.e. building the DataLoaders. 

### Creating DataLoaders
To create the DataLoaders I used two packages one is `PyTorch` and the other is `PyTorch Lightning`. The dataset was first split into tjhree sets `train`, `val` and `test` with a $90:10:10$ split with stratification. While building the dataloader there are a few things that I did with the images in order to make it compatible for processing by the latest vision models.
* The images have a size of $(H, W)$, which is incompatible for processing. Therefore I added a new dimension to the image using `np.expand_dims(...)` thereby converting the shape to $(H, W, 1)$ where the last dimension represents the number of channels.
* Next, I concatenated my image data $3$ times along my channel dimension so that I have an array of shape $(H, W, C)$ where $C=3$ . This process is used to replicate my set of images as an RGB image.
* Next, I resized the image to $(256, 256)$
* Seperate sets of transformations were applied to `train` and `val/test` sets:
	* For `train` set - transformations such as `RandomBrightnessContrast()`, `GaussianNoise()`, `HorizontalFlip()`, etc. were implemented.
	* For `val/test` set - only `CenterCrop(224, 224)` and `Normalize(224, 224)` was applied.
* After applying the transformation I then swapped back my channel dimension to the first dimension i.e. converted my array shape from $(H, W, C)$ to $(C, H, W)$.

Since my DataLoaders are ready it is time for me to build my model.

### Creating Classification Model
![[Task-5 Model.png]]
Since the objective of this task is to explore transformers, I went with a ViT model with patch size of $16$ and image dimension of $224 \times 224$. 
The model class was created using `PyTorch Lightning` over vanilla `PyTorch`.

### Training the Model

The model is trained using the *binary cross-entropy loss*:
$$
\begin{equation}
L = -\sum_{i=1}^n (y_i\log(\hat{y_i}) + (1- y_i)\log{(1 - \hat{y_i})})
\end{equation}
$$

For training the model I used **Automatic Mixed Precision** by setting the precision level to $16$. I also used **Early stopping** with the objective to monitor the *validation AUCROC* score with a patience level of $3$ to avoid model overfitting. *Model Checkpointing* was used to checkpoint the top-1 models by monitoring the *validation AUCROC* scores. 
The `max_epochs` was even though set to $10,000$, the training stopped early because of *early stopping*.

### Testing the Model

After training the model, the final stage was testing it on the test set created before. For testing the model I used the saved checkpoint of my model and the result obtained is as follows:
| metrics  | values |
|----------|--------|
| Accuracy | 0.95   |
| F1Score  | 0.95   |
| AUCROC   | 0.99   |
