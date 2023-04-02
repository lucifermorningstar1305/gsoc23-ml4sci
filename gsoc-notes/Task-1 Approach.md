
## Problem Overview

Task-1 deals with classifying images into three categories:
* No Substructure
* Subhalo Substructure
* Vortex Substructure

The **Evaluation metric** for this task is the AUCROC (Area under the Receiver Operating Characteristic curve).

## How to Solve the problem

Well looking at the above problem description it is clear that we need to build a 
classification algorithm, but before that we need to explore the data first. 

### Exploring the Target Distribution
![[Screenshot from 2023-04-03 01-27-48.png]]
Looking at the above plot it is clear that the data has each targets **equally distributed**. Still, it is a good idea to do stratification when splitting the data.

### Exploring the Image Data
![[Screenshot from 2023-04-03 01-34-44.png]]
While working with the images I noticed that the images where already shaped as $(C, H, W)$ where $C$ represents the number of channel, $H$ represents the height of the image and $W$ represents the width of the image. But the value of $C$ was $1$ for every image this is because the images are grayscale images. 

Also the values of the images were already normalized, meaning the values belonged in the range of $[0, 1]$.

Now moving onto to the next step of the approach, i.e. building the DataLoaders. 

### Creating DataLoaders

To create the DataLoaders I used two packages one is `PyTorch` and the other is `PyTorch Lightning`. The training dataset was first split into two sets `train` and `minival` with a $90:10$ split with stratification. While building the dataloader there are a few things that I did with the images in order to make it compatible for processing by the latest vision models.
* First I swapped the channel dimension to the last i.e. converted the images from $(C, H, W)$ to $(H, W, C)$.
* Next, I concatenated my image data $3$ times along my channel dimension so that I have an array of shape $(H, W, C)$ where $C=3$ . This process is used to replicate my set of images as an RGB image.
* Next, I resized the image to $(256, 256)$
* Next I applied different set of transformations for my training and validation data.
	* For training - `RandomBrightnessContrast()`, `Rotate()`, `CenterCrop(224, 224` and `Normalize()` was applied.
	* For validation - `CenterCrop(224, 224)` and `Normalize()` was applied.
* After applying the transformation I then swapped back my channel dimension to the first dimension i.e. converted my array shape from $(H, W, C)$ to $(C, H, W)$.

Since my DataLoaders are ready it is time for me to build my model.

### Creating Classfication Model
![[Task-1 Model.png]]
For our classification algorithm, I went with the **EfficientNet-b2** model as our image feature extractor and then added a custom classification layer with $3$ neurons that represents each of three classes. 
To build the model I used `PyTorch Lightning` over vanilla `PyTorch` as it wraps a lot of functions and makes the code cleaner/tidier. 

### Training the Model

The model is trained to minimize the cross-entropy loss:
$$
\begin{equation}
L = -\sum_{i=1}^n y_i\log{\hat{y_i}}
\end{equation}
$$
For training the model I used **Automatic Mixed Precision** by setting the precision level to $16$. I also used **Early stopping** with the objective to monitor the *validation AUCROC* score with a patience level of $3$ to avoid model overfitting. *Model Checkpointing* was used to checkpoint the top-1 models by monitoring the *validation AUCROC* scores. 
The `max_epochs` was even though set to $10,000$, the training stopped early because of *early stopping*.

### Testing the Model

After training the model, the final stage was testing it on the validation set provided by the competetion. For testing the model I used the saved checkpoint of my model and the result obtained is as follows:

| metrics  | values |
|----------|--------|
| Accuracy | 0.93   |
| F1Score  | 0.93   |
| AUCROC   | 0.99   |

