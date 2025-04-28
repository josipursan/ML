# Learning rate warm-up  

Why does LR warm-up help?  
-stabilizes weight initialization - at the start of training model's weights are most often randomly initialized  
&nbsp;&nbsp;&nbsp;-at the very start of training, a large LR value can lead to drastic weight changes, thus causing oscillatory behaviour  
&nbsp;&nbsp;&nbsp;-warm-up allows the network to ease into the training process using smaller LR updates  
  
-enables a smoother gradient flow - large initial LR can result in exploding gradients  
&nbsp;&nbsp;&nbsp;-starting with a small initial LR, and then gradually, patiently, increasing it is supportive of a well behaved model  
  
-prepares for complex layers - architectures that include batch normalization layers rely on stable statistics, which take time to settle, which is where LR warm-up helps  
  
-supports large-scale models - LR warm-up is particularly useful in large models because it reduces the possibility of early-stage training oscillations caused by unstable gradients  
  
-high learning rates are recommended mostly for smaller, simpler models  
