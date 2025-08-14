# W2  
## Classic networks  
https://www.youtube.com/watch?v=dZVkygnKh1M&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=13  
  
-professor here does a nice job of provoding an overview that is through enough, yet not too in depth  
-LeNet 5, AlexNet, VGG-16  
  
## Resnets  
https://www.youtube.com/watch?v=ZILIbUvp5lk&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=14  
  
Very deep networks are sometimes difficult to train because of vanishing and exploding gradients.  
Skip connections allow us to take activation from one layer and jump to some very deep layer, skipping all layers in between.  
  
Resnets - Residual networks  
  
-resnets are built ouf of *residual blocks*  
  
-consider a conventional network - in order to reach layer *l+1* you need to apply some linear operation to activation from layer *l* ($a^{[l]}$), and then apply ReLU nonlinearity, giving us $a^{[l+1]}$  
&nbsp;&nbsp;&nbsp;-now we again linear operation to $a^{[l+1]}$ and then ReLU nonlinearity, finally giving us $a^{[l+2]}$  
&nbsp;&nbsp;&nbsp;-this path is called the **main path** :  
<p style="text-align: center">
    <img src="../screenshots/MainPat.png"/>
</p>  
  
-instead of traversing the main path we introduce a **short path** that jumps straight from $a^{[l]}$ to point before last ReLU call is made :  
<p style="text-align: center">
    <img src="../screenshots/ShortPath.png"/>
</p>  
  
-the idea of resnets is that we want to keep the network, while it is learning, aware of the original input signal  
&nbsp;&nbsp;&nbsp;-keep in mind that the input signal is very quickly driven to unrecognizability due to the applied activation functions  
  
## Why Resnets work  
https://www.youtube.com/watch?v=RYth6EbBUqM&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=15  
  
