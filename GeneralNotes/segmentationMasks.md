# Segmentation masks  
https://blog.roboflow.com/how-to-create-segmentation-masks-with-roboflow/  
https://www.cloudfactory.com/blog/segmentation-mask-visualization-bringing-clarity-to-ai-models  
  
-segmentation mask - a specific portion of image that is isolated from the rest of the image  
&nbsp;&nbsp;&nbsp;-it is a representation that highlights how an image is broken up into regions based on object boundaries or features  
&nbsp;&nbsp;&nbsp;-there are several CV techniques that generate a segmentation mask :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-**semantic segmentation** - this segmentation model involves identifying and labeling each pixel of an image; it uses deep learning to classify pixels based on shared features  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-**instance segmentation** - a combination of object detection and semantic detection; it only labels each pixel with a class, while also distinguishing between different instances of the same object; is especially beneficial in scenarios where distinguishing between similar objects is critical  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-**panoptic segmentation** - combines semantic and instance segmentation, providing a more detailed understanding of a scene; assigns each pixel a class label and an instance ID, producing a comprehensive representation of an image   