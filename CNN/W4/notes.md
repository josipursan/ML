# W4  
## What is face recognition?  
https://www.youtube.com/watch?v=-FfMVnwXrZ0&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=32  
-verification - you input an image and a name/ID  
&nbsp;&nbsp;&nbsp;-the system should output whether the inputted image matches that of the name/ID  
  
-recognition - you input an image  
&nbsp;&nbsp;&nbsp;-the system has a database of K persons  
&nbsp;&nbsp;&nbsp;-output ID if the image belongs to any of the K persons in the database  
  
-recognition is harder because its accuracy must be very high  
  
## One-shot learning  
https://www.youtube.com/watch?v=96b_weTZb2w&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=33  
  
-for most face recognition applications you need to be able to recognize a person using one image/one example  
  
-a system has only one picture of each person in its database and needs to be able to let that person through  
  
-instead of training a DNN using only one example per person, and getting a NN that performs veeeeery poorly, we will learn a **similarity** function  
  
-a similarity function, d(img1, img2), compares two images, img1 being the input from a camera on the enterance, and img2 being image stored in database  
-for each pair of images similarity function will output the degree of difference between these two images  
-if the difference is **less** than some value ($\tau$), we can conclude the images match (ie. same person is in both images)  
-if the difference is greater than some value ($\tau$), we can conclude the images don't match (ie. they show different people)  
  
## Siamese network  
https://www.youtube.com/watch?v=6jfw8MuKwpI&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=34  
  
-imagine you have an ordinary network, except you ignore the usually used softmax output layer that is used to determine 