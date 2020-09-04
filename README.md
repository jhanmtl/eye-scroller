This project is a demonstration in using React with TensorFlowJS to perfrom realtime eye-detection and landmark-extraction via webcam and
client-side inference. Result is an intelligent interface for controlling the scrolling action of a document by the user's blinking action. Inspired by the difficulty of
simply turning a page and reading independently experienced by people suffering from whole-body paralysis. 

[Try the live demo yourself here at this link](https://jhanmtl.github.io/eye-detector/) (webcam access required)
Example shown below

![](./public/demo.gif)

* 60+  fps on GPU
* 5-10 fps on CPU



Customized SSD model originally implemented in TensorFlow Python. Specialized for eye detection. 

Upon training completion, model converted to TensorflowJS and integrated with React for realtime inference with client browser and webcam.

Utilizes Functional Architecture from TensorFlow Python and custom training routines based on tf.GradientTape. Training data obtained from the 
Landmark guided face Parsing ([LaPa](https://github.com/JDAI-CV/lapa-dataset)) dataset by JDAI-CV and preprocessed with on-line augmentation via tf.data API.

Minimal model weights (2mb) with only one prior per feature location. Based on the 14x14 feature
output map of MobileNetV2, downsampled with additional conv2D layers until final feature map of size 6x6. 
See model schematic below.


![](./public/modelPlot.png)




