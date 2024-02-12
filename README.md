# CSE151A-Project

## Data Exploration and Preprocessing
Here's a link to my noteboook for data processing [Notebook here](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/DataExploration.ipynb)
### Labeled Data
My labeled data was created by myself by scraping images of keyboards from popular mechanical keyboard vendors like KBDFans, and manually drawing bounding boxes over each keycap in the image. Here's an example: ![Annotation Image](https://github.com/portoaj/CSE151A-Project/raw/main/Examples/dataannotationexample.png)

Each of those neon yellow boxes around the keycaps were those bounding boxes that I manually drew. I draw bounding boxes for around a hundred keycaps each for tens of images, which was pretty painstaking but should allow me to train an object detection model capable of finding the keycaps within an image.

Each image was preprocessed to be 640x640, as this is the standard for YOLO models, and has worked well for me in the past. To maintain the aspect ratio the image was stretched and then had white edges added as needed to make each image a 640x640 square. Here's an example: ![Data preprocessing example](https://github.com/portoaj/CSE151A-Project/raw/main/Examples/imageresizing.png)

To augment my data I used a combination of image level and object level data augmentation tools that are built in to roboflow. While I could have done the image level augmentation like changing the saturation myself, the object level augmentation would have been very challenging because augmentations like shearing the image would also move where the labels have to be.

After looking through the different options, here is what I settled on:

Rotation between -3 and +3 degrees as the images may not always be perfectly oriented upwards. I decided against 90 degree rotations as I think it's reasonable to expect my input images to be facing up.

Shear +- 5 degrees horizontally and vertically. If the image isn't aligned perfectly or there is some distortion to the input image, this should help make the image generalize to that.

Saturation +- 10% as some input images may appear more saturated than others depending on the keyboard colors and the camera settings.

Blur of up to a 1px radius as parts of the keyboard could be blurry depending on how the focus on the camera is set.

Further exploration of my labeled data is in the notebook [Here](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/DataExploration.ipynb)
### Unlabeled Data
I'm not sure if I'm going to be able to use the unlabeled data later in the project as training object detection models from unlabeled data is very challenging, but it will be an interesting exercise to collect it.

To do this I'm going to scrape a popular mechanical keyboard vendor, kbdfans.com. I'm registered as an affiliate with KBDFans as part of my kbdlab.io project and they've given me permission to use their images as part of that program.

One trick I'm going to use to make my web scraping simpler is to use the automatic API generated by they website through shopify. Here's the documentation: https://shopify.dev/docs/api/ajax/reference/product. Interestingly, shopify automatically generates this API for all of their websites, likely to make scraping simpler so that people don't overwhelm their websites with scraping requests.

The code for the scraping is in the notebook [Here](https://raw.githubusercontent.com/portoaj/CSE151A-Project/main/DataExploration.ipynb) 
