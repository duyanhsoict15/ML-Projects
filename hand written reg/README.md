# Handwritten
With Deep learning, the handwriting recognition problem is not too new, with the data of 26 English letters on Kaggle, we can completely build a training model to recognize 26 classes (corresponding to with 26 letters). In this project, I will perform letter prediction by writing directly with the Gradio library.

<p align="center">
	<img src="https://github.com/duyanhsoict15/ML-Projects/blob/main/hand%20written%20reg/imgTest/handwritten.gif" />
</p>

# How it work
## Part 1: Training the model
- Step-by-step training details at [TrainingModel.ipynb](https://github.com/duyanhsoict15/ML-Projects/blob/main/hand%20written%20reg/TrainingModel.ipynb)
## Part 2: Making predictions
With [Gradio](https://gradio.app), it provides us with the fastest and most convenient way to deploy machine learning model predictions. In this project, with input being a handwritten letter Gradio provides us with a method called Sketchpad, which helps us draw handwritten letters in a very convenient way.

<p align="center">
	<img src="https://github.com/duyanhsoict15/ML-Projects/blob/main/hand%20written%20reg/imgTest/3.png" />
</p>

First we need to declare the model we have trained:

```python
from tensorflow.keras.models import load_model
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
model = load_model('modelHandWritten.h5')
```

Process input and predict results:

```python
def classify(img):
	img_final = cv2.resize(img, (28, 28))
	img_final = np.reshape(img_final, (1, 28, 28, 1))
	prediction = model.predict(img_final).flatten()
	return {word_dict[i]: float(prediction[i]) for i in range(25)}
```

Display:

```python
iface = gr.Interface(
	classify,
	gr.inputs.Image(shape=(224, 224), image_mode='L', invert_colors=True, source="canvas"),
	gr.outputs.Label(num_top_classes=3),
	capture_session=True,
	)
```

If we want it to create a link for online prediction, we need to set: `share=True`

```python
if __name__ == "__main__":
	iface.launch(share=True)
```
