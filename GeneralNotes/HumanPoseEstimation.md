# Human pose est mation  
https://www.v7labs.com/blog/human-pose-estimation-guide  
  
-there are three approaches to modeling the human body :  
&nbsp;&nbsp;&nbsp;-skeleton based model  
&nbsp;&nbsp;&nbsp;-contour-based model  
&nbsp;&nbsp;&nbsp;-volume-based model  
  
-HPE (Human Pose Estimation) applications and solutions are mainly found in the field of computer vision  
-HPE can be solved using a classical approach or deep learning  
  
## Classical approaches to 2D HPE  
-the term *classical approaches* mainly refers to methods involving *swallow machine algorithms*  
-early HPE work included the implementation of a random forest algorithm within a *pictorial structure framework*  
  
-pictorial structure framework (PSF) is commonly referred to as one of the traditional HPE methods  
-two components make up PSF :  
&nbsp;&nbsp;&nbsp;-discriminator - models the likelihood of a certain body part being present in a certain location; in other words, it locates body parts  
&nbsp;&nbsp;&nbsp;-prior - models the probability distribution over pose using the prior output from the discriminator, the modeled pose should be real; it basically figures out how each body parts should be positioned  
  
-the PSF's objective is to represent the human body as a collection of coordinates for each body part in a given input image  
-PSF works well when the image has clear and visible limbs  
-however, it falls short if the limbs are hidden by the body positioning, or not sufficiently visible due to angle  
  
## Deep learning based approaches to 2D HPE  
-using CNN's is generally speaking a better approach for a lot of problems (classification, detection, segmentation)  
-CNN's are awesome at extracting patterns, given enough data  
  
## Human pose estimation using Deep Neural Networks  
-DNNs (Deep Neural Networks) are very proficient at estimating the body pose of an individidual, however they struggle with multiple humans in a single scene  
-more people in a single image means more different points to estimate, more different positions, more different interactions, all of which can lead to increased inference times  
  
-to tackle this multi-person issues, two approaches have been recommended :  
&nbsp;&nbsp;&nbsp;-**top-down** : first localize the humans in the image/video, and then estimate the parts, followed by pose computation  
&nbsp;&nbsp;&nbsp;-**bottom-up** : first estimate the human body parts followed by computing the pose   
  
-note that it was mentioned the *top-down* approach generates a lot of errors in localization, as well as inaccuracies during prediction
  
-several interesting algorithms for multip-person HPE are mentioned :  
&nbsp;&nbsp;&nbsp;-OpenPose  
&nbsp;&nbsp;&nbsp;-AlphaPose (RMPE - Regional MultiPerson Estimation)  
&nbsp;&nbsp;&nbsp;-DeepCut  
&nbsp;&nbsp;&nbsp;-Mask R-CNN  
  
-Mask R-CNN - very popular algorithm for instance segmentation  
&nbsp;&nbsp;&nbsp;-simultaneously localizes and classifies objects by creating a bounding box around the object  
&nbsp;&nbsp;&nbsp;-basic architecture can easily be tuned for HPE  
&nbsp;&nbsp;&nbsp;-CNNs are used to extract features and representation from the given input  
&nbsp;&nbsp;&nbsp;-extracted features are then used to propose where the object might be present using a **Regional Proposal Network** (RPN)  
&nbsp;&nbsp;&nbsp;-a layer called **RoiAlign** is used to normalize the extracted features so that they are all uniformly sized  
&nbsp;&nbsp;&nbsp;-extracted features are passed into the parallel branches of the network to refine the proposed regions of interest (RoI) to generate bounding boxes and the segmentation masks  
&nbsp;&nbsp;&nbsp;-the mask segmentation output can be used to detect humans in the given input  
&nbsp;&nbsp;&nbsp;-
  
