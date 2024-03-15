1.A complete Introduction
# Final Project Writeup

## Previous notebooks
[Data exploration & preprocessing notebook](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/DataExploration.ipynb)
[Model 1 notebook](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/CSE151AProjectMilestone3.ipynb)
[Model 2 notebook](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/CSE151AProjectMilestone4.ipynb)
[Model 3 notebook](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/CSE151AProjectMilestone5.ipynb)

## Introduction
The basis of my project revolves around my website [kbdlab.io](https://kbdlab.io). It's a platform for people to create custom mechanical keyboard builds by choosing from different keyboard kits, switches, and keycap sets which they can then build or share with others. One piece of information that I wanted to display on my website was the number of keycaps in a keycap set or the number of keys supported by a keyboard kit. However, finding this information for the thousands of keyboards and keycap sets that exist by hand would be a painstaking process, so I wanted to train an object detection model to find all of the keycaps in an image.

![KBDLab Keyboard Build (fig 1
)](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/kbdlabbuild.png)
*KBDLab Keyboard Build*

I hope to use this model to enhance my website and give my thousands of users more free information about the keyboards hobby.

## Methods
### Data Exploration
[Data exploration & preprocessing notebook](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/DataExploration.ipynb)
I created my own dataset which can be found [here](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/AnnotatedData.zip) in COCO format and [here](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/AnnotatedDataYoloFormat.zip) in YOLOv8 format. I created this dataset by scraping images from kbdfans.com, who I'm affiliated with and then labeling them using the Roboflow platform. In the end I had labeled 36 images with 1,713 keycap annotations.
![Example of keycap annotations ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/dataannotationexample.png)
*Example of keycap annotations*


I also demonstrated how to collect unlabeled data by scraping a website that conformed to the Shopify API standard:
```
import os
import requests
import tqdm as tqdm
# We can find all the different product collections at kbdfans.com/collections.json
collections = requests.get('https://kbdfans.com/collections.json?limit=250').json()['collections']
print([collection['handle'] for collection in collections])

# Heres the data for their different keycap sets
keycap_sets = requests.get('https://kbdfans.com/collections/keycaps/products.json?limit=250&page=0').json()['products']
print([product['title'] for product in keycap_sets])

if not os.path.exists('UnlabeledData'):
  os.mkdir('UnlabeledData')

# Image downloading code modified from https://stackoverflow.com/questions/30229231/python-save-image-from-url
for i, keycap_set in tqdm.tqdm(list(enumerate(keycap_sets))):
  image_url = keycap_set['images'][0]['src']
  img_data = requests.get(image_url).content
  with open(f'UnlabeledData/{i}.jpg', 'wb') as handler:
      handler.write(img_data)
```
Using that script I was able to download 179 different keycap images.

### Data Preprocessing
[Data exploration & preprocessing notebook](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/DataExploration.ipynb)

Later I switched to my own data augmentation workflow, but for this step I used Roboflows built-in data processing and augmentation workflow. I scaled all of my images to 640x640 with white borders included as necessary. Here were the data augmentations I used:
 Rotation: Between -3° and +3°
Shear: ±5° Horizontal, ±5° Vertical
Saturation: Between -10% and +10%
Blur: Up to 1px

Using these settings I tripled my dataset to 108 images with 5,139 annotations.

I also demonstrated how I would've done my own image preprocessing and augmentation on the unlabeled data at the beginning of my milestone 3 notebook.

Here's the preprocessing:
```
# We load and preprocess our images
import numpy as np
from PIL import Image
from tqdm import tqdm
preprocessed_images = []
for image_path in tqdm.tqdm(os.listdir("/content/UnlabeledData")[0:160]):
  img = Image.open('/content/UnlabeledData/'+ image_path)
  # We will skip the few images that aren't square
  if img.width != img.height:
    continue
  img = img.convert('RGB')
  img = img.resize((512,512), resample=Image.Resampling.LANCZOS)

  preprocessed_images.append(np.array(img, dtype=np.uint8))
preprocessed_images = np.stack(preprocessed_images)
```

Here's the data augmentation with differences in contrast, darkness, scale, rotation, and shearing:
```
import imgaug as ia
import imgaug.augmenters as iaa

print(preprocessed_images.shape)

seq = iaa.Sequential([
    iaa.LinearContrast((0.9, 1.1)),
    iaa.Multiply((0.95, 1.05)),
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-8, 8),
        shear=(-6, 6)
    ),

])
result = None
for i in range(2):
  images_aug = seq(images=preprocessed_images)
  if result is None:
    result = images_aug
  else:
    result = np.concatenate((result, images_aug))
```
### Model 1
I decided to try out the HuggingFace platform to use SOTA object detection models such as DETR.

My training consisted of finetuning a pre-trained model from the HuggingFace database:
```
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

I had to load my dataset in a way that conformed to DETR model:

```
def generate_dataset(image_ids):
  dataset = []
  for image_id in image_ids:
    image_obj = list(filter(lambda image: (image['id'] == image_id), images))[0]
    #print(image_obj)
    relevant_annotations = list(filter(lambda annotation: (annotation['image_id'] == image_id), annotations))

    objects = {'id': [annotation['id'] for annotation in relevant_annotations], 'area': [annotation['area'] for annotation in relevant_annotations], 'bbox':[annotation['bbox'] for annotation in relevant_annotations], 'category':[annotation['category_id'] for annotation in relevant_annotations]}

    dataset.append({'image_id': image_id, 'image': Image.open('AnnotatedData/' + image_obj['file_name']), 'width': image_obj['width'], 'height': image_obj['height'], 'objects': objects})
  return dataset

raw_train_dataset = generate_dataset(range(int(len(images)*.7)))
raw_validation_dataset = generate_dataset(range(int(len(images)*.7), int(len(images)*.8)))
raw_test_dataset = generate_dataset(range(int(len(images)*.8), len(images)))

assert len(raw_train_dataset) + len(raw_validation_dataset) + len(raw_test_dataset) == len(images)
```

Here are the arguments I specified when loading the model:

```
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label={0:'background', 1: 'keycap'},
    label2id={'keycap': 1, 'background': 0},
    ignore_mismatched_sizes=True,
)
```

There's a lot more code for further transforming my data into a DETR compatible format in the notebook. Of particular interest are the data augmentations I performed involving horizontally flipping the image, adjusting aspects of the image coloration, and simulating lossy image scaling:
```
transform = albumentations.Compose(
    [
        albumentations.HorizontalFlip(p=0.1),
        albumentations.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=.5),
        albumentations.Downscale(scale_min=.15, scale_max=.15)

    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)
```

Here are the hyperparameters I decided on for the model:
```
training_args = TrainingArguments(
    output_dir="detr-resnet-50",
    per_device_train_batch_size=8,
    num_train_epochs=100,
    fp16=False,
    save_steps=100,
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    evaluation_strategy='steps'
)
```
The model trained for a total of 100 epochs and 400 steps.


### Model 2
This model was mostly the same as Model 1 except that I used a Deformable DETR model instead of a DETR model.

Here were my training arguments:
```
training_args = TrainingArguments(
    output_dir='deformable-detr',
    per_device_train_batch_size=4,
    num_train_epochs=100,
    fp16=True,
    save_steps=20,
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    evaluation_strategy='steps'
)
```

### Model 3
For model 3 I used a YoloV8 model using Ultralytics. I reexported my dataset in a YoloV8 compatible format using Roboflow.

Training a model with Ultralytics was very terse:
```
from ultralytics import YOLO
# Code sourced/ modified from the github page: https://github.com/ultralytics/ultralytics
# And from the docs: https://docs.ultralytics.com/modes/train/#train-settings

model = YOLO("yolov8s.pt")
model.train(data="/content/AnnotatedDataYoloFormat/data.yaml", epochs=100, batch=8, plots=True, seed=0, patience=25)
```

Finally I generated the images for what the testing output looked like:
```
# Gettings metrics and results for the testing set
model.val(split='test', plots=True)
# Show the validation results
display(Image.open('/content/runs/detect/train2/val_batch0_pred.jpg'))
```

## Results
### Data Exploration
The following graph shows the number of keycaps annotated in each image:
![Number of annotations by image ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/numannotations.png)
*Number of annotations by image*
There was a mean of 47.58 keycaps in each image with a standard deviation of 39. The image with the least number of keycaps had 4 and the image with the most keycaps had 166.
### Data Preprocessing
In the end I generated a zip file containing 179 unlabeled images, which I didn't end up using to train my models.
Here's an example of one of the images
![Image genereated by preprocessing](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/unlabeleddataexample.jpg)
*Image genereated by preprocessing*
### Model 1
![Training/Validation loss for model 1 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/model1graph.png)
*Training/Validation loss for model 1*

![Testing metrics for model 1 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/project1eval.png)
*Testing metrics for model 1*

Finally, I ran the model and displayed the inferences on one of my testing images:
![Testing image with annotations for model 1 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/project1result.png)
*Testing image with annotations for model 1*

### Model 2
![Training/Validation loss for model 2 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/model2graph.png)
*Training/Validation loss for model 2*

![Testing metrics for model 2 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/model2eval.png)
*Testing metrics for model 2*

Here are a couple examples of inference with the model.

![First testing image with annotations for model 2 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/model2result1.png)
*First testing image with annotations for model 2*

![Second testing image with annotations for model 2 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/model2result2.png)
*Second testing image with annotations for model 2*

### Model 3

![Training/Validation loss for model 3 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/model3graph.png)
*Training/Validation loss for model 3*

![Testing metrics for model 3 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/model3eval.png)
*Testing metrics for model 3*

![Testing image with annotations for model 3 ](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/model3result.png)
*Testing image with annotations for model 3*

## Discussion

### Data Exploration
The data exploration phase of the project was mostly unsurprising to me since I made my own dataset, meaning I'd already seen every image. The one surprising thing was the amount of variation in the number of keycaps between images. The standard deviation in the number of keycaps was 39. This surprised me because I thought they'd be pretty similar but looking back it makes sense since some of the images just had something like a numpad with 15 or so keys versus a full keyboard at over 100. This could definitely make training a bit harder since the keyboards and key layouts will be different in different images. Additionally, an image with few keycaps might have them bigger in the image than a keyboard with hundreds of keycaps that all need to fit into an image. That scale difference may have further complicated training.
### Data Preprocessing
Labeling all the images took about 6 hours and I was pretty reluctant to spend time labeling any more images. Even though I had very few images (especially after splitting the data), I believed that my high number of annotations would be sufficient to train a decent model. Initially I also used Roboflow to triple the number of images/ annotations I had using their data augmentation tools but I later chose to do the data augmentation myself and got rid of the augmented data in my original dataset.


### Model 1
In the past I've only used traditional CNN methods for Object Detection like SSDs, YOLO and Faster-RCNN so I wanted to try out a more modern Transformer-based technique. Since DETR had some impressive COCO benchmark scores and was available on Hugging Face I decided to give it a try.

I had to specify a few arguments when loading the model such as giving the model two dictionaries that linked my labels(background or keycap) with an id number. More importantly, I had to specify ignore_mismatched_sizes to true since the model had been pretrained on a dataset with 100 classes, whereas I was only using 2.

Albumentations was an image augmentation library I used that preserved the object detection labels even when scaling, rotating or moving the image which was a lot easier than I thought it would be. 

I decided on occasionally horizontally flipping the image, but with a low probability since keyboard images are normally left to right and I wanted the model to have that context. I decided to add color jitter through HSL and contrast on half of the images so my model would be more robust to inputs with different colors. Finally, I added a downscale and reupscale effect half the time to make my model more robust to low resolution or compressed photos.

My training arguments were pretty standard and mostly left to the default from the Hugging Face docs. I did a lot of experimentation to find the number of epochs(100) that would result in the model being fully trained without wasting time training for lots of extra epochs. I also disabled fp16 quantization as it was causing me some errors during model evaluation.

The results were somewhat dissapointing for this model even though it seemed to converge properly on the training/validation loss graph. Since I wanted to know how many keycaps were in an image recall was really important to me and with maxDets=100 and area=all which was the most realistic measure of recall, my recall was only 0.447. I was aiming for a recall of at least 0.8 with this project. Additionally, the example that I used to show the inference had many labels on the same keycap and then some keycaps with no labels, so there was clearly room for improvement.

One final issue with DETR models for my use case was that by default they supported only outputting 100 labels in an image. However, my use case could sometimes have huge sets of keycaps with much over 100 labels in a single image. I tried increasing a hyperparameter of the model to enable more labels to detected in an image, but this resulted in having to drop the pretrained weights, and without those my model wouldn't converge at all, so I left the max number of predictions at 100.

### Model 2
I chose this model partially for convenience since I was optimistic that it would be easy to train in a similar fashion to my previous DETR model. After a few hours of troubleshooting I was able to get it to run on nearly the same code. I think that in a project where you're evaluating many different models, it's important to be able to reuse common code and work wherever possible to limit the complexity of the project.

One reason I was optimistic about this model was that from what I'd read the Deformable DETR model converged faster than its vanilla DETR counterpart, and since my training set was so small, I thought this could help it learn my training set better.

A surprising thing when I trained this model was that my eval_loss was actually lower than my training_loss. I'm pretty confident this wasn't due to data leakage, and even if it was you'd expect the eval_loss and training_loss to be about equal. I think this was because the training set was about 25 images while the validation set was only around 5 images so I think the model just happened to get a validation set that it was well-trained for relative to the average image in the training set.

I had the same issue with model 1 where setting the maximum number of predictions to anything but 100 would remove the pretraining and cause my model to fail to converge.

My training arguments were actually pretty important when training this model as I ran out of VRAM on the T4 GPU I was using with Google Colab. In order to train the model, I had to lower the batch size to 4 from 8 and reenable the fp16 quantization. With those changes, my model was trained similarly to my first one.

I was much happier with my results for model 2 than my results for model 1. My average recall for area=all and maxDets=100 shot up to 0.662 from 0.447 before which was a big improvement and you could definitely see this in the inferences I generated from the testing set. The first image on the numpad is very rough, and it looks like the model was scared to label most of the keycaps as they looked pretty different from the keycaps in the training set I'd imagine with the non-standard characters on them. The second inference image looked very promising though. Aside from adding a keycap label to the charger, not labeling the keyboard, and missing f8, it labeled every other keycap in the second image correctly.

This example really showed me that the task I wanted to train the models on was possible, but it was possible that I'd need more training data with unusual examples like in the first testing image.

A final oddity with this model is that unlike with the DETR model it didn't only guess keycap or background from the dictionaries I provided, sometimes it would guess other classes like computer mouse or fan that it had learned in its pretraining and I had to write code to remove those predictions.
### Model 3
For this model I was planning on using my existing codebase but trying a smaller YoloS model to see if a smaller model would better learn my relatively simple object detection task but there were a lot of issues trying to train the YoloS model with Hugging Face and my existing codebase. It ended up being a lot easier to switch to Ultralytics for this model since their API is very abstracted out.

In the end I used a YoloV8-Small model for my third model. Looking at their graphs of relative model performance it looked like the smallest model, nano, might be too inaccurate. The accuracy jump to small looked promising so I used it while the other models had more modest accuracy jumps and would've been a lot more complex, so I was worried they wouldn't work as well with my small training set.

![Graph of different Yolo models performance (source Ultralytics)](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/Examples/ultralyticsmodels.png)
*Graph of different Yolo models performance (source Ultralytics)*

The YoloV8-S model was surprising good out of the box. I did adjust some hyperparameters, but more for convenience than performance as my first attempt at training had astonishing results as I'll discuss later. In particular, I set plots to true so that it would generate my training/ validation loss/ metric plots, set the seed to 0 for consistency, and set the patience to 25 so that the model would early stop if there were no gains in performance after a while.

The evaluation metrics were pretty incredible, some of the training epochs even had a validation recall of 1.0. Considering that was on hundreds of labels, that's quite impressive. On my testing set, I got a recall of 0.987 and a precision of 0.978. Of course these numbers come from a very small testing set of only 3 images unfortunately.

Viewing the inferences on the test images showed that the only errors occurred on the spacebar of the second test image, which makes sense since space bars are much wider than most keys. I'm confident that with more training data, especially with more spacebars and unique looking keycaps, the model would be almost perfect.

## Conclusion
The course of this project taught me a lot about the practical portions of deploying ML models to a real-world tasks. In particular, that the vast majority of your time is going to be spent on getting good data, formatting it correctly, and debugging the inputs to your model, whereas relatively little time is spent on the actual model portion itself.

It also showed me that avoiding complexity with your model is ideal, unless your dataset is complicated enough that a complex model is necessary. The SOTA DETR models based on massive transformer architectures and trained on multiple GPUs ended up performing significantly worse than a small YOLO model, because the task was better suited to a less complex model.

This project also showed me the importance of having enough data. Both my DETR models did decently well on a standard image of a keyboard with keycaps that had standard characters, but when you threw in something somewhat abnormal like different characters, sizing, or layout, the model would suddenly perform horribly. I think the best solution to this would simply be to have a wider array of data so that the model could be trained to account for these situations.

That being said, I also learned how painful it is to actually collect good data. My relatively small dataset took me hours of dragging 1,700 boxes, so creating a massive dataset like COCO must be a masssive effort.

With all that being said, I consider my project to be a success and the YOLO model that I ended up with seems ready for me to deploy to production in the future in order to count the number of keycaps on a given keycap set or keyboard, even if it could still benefit from some more training data. 
## Collaboration
Andrew Masek: Completed the entire project