This is a demonstration in using React with Tensorflow to perfrom realtime eye-detection and landmark-extraction via webcam and
client-side inference. 

It is an end-to-end Machine Learning project that spans data collection, data augmentation, model design, customized training,
model optimization, and deployment via Tenosrflow Python --> Tensorflow JS convesion and React frontend integration. 

Result is an intelligent interface for controlling the scrolling action of a document by the user's blinking action. Inspired by the difficulty of
simply turning a page and reading independently experienced by people suffering from whole-body paralysis.

[Try the live demo yourself here at this link](https://jhanmtl.github.io/eye-scroller/) (webcam access required)
* 50+  fps on GPU
* 10-20 fps on CPU

Example shown below. Models details follow.
![](./public/demo.gif)

Two models trained and deployed. Both utilize Functional architecture from TensorFlow Python. Training data obtained from the 
Landmark guided face Parsing ([LaPa](https://github.com/JDAI-CV/lapa-dataset)) dataset by JDAI-CV and preprocessed with on-line augmentation via tf.data API.

1. Customized SSD model originally implemented in TensorFlow Python. Specialized for eye detection. Minimal model weights (2 mb) with only one prior per feature location. Based on the 14x14 feature
output map of MobileNetV2, downsampled with additional conv2D layers until final feature map of size 6x6. Custom training routines based on tf.GradientTape.
See model schematic in modelSchematics folder.

2. Customized MLP, specialized for detecting 4 landmarks on the upper and lower eye lids.
Built on top of the first few layers (up to the 'conv2_block2_0_relu') of a Densenet121 feature extractor. Extremely lightweight (564 kb).
see model schematic in modelSchematics folder.

Upon training completion, model converted to TensorflowJS and integrated with React for realtime inference with client browser and webcam.




