This project is a demonstration in using React with TensorFlowJS to detect eyes via webcam.

Original model implemented in TensorFlow Python. Specialized for eye detection with 3 priors per feature location on the 7x7 final output of a MobileNetV2
feature extractor. Utilizes Functional tf architecture and custom training routines based on tf.GradientTape. Training data obtained from the 
Landmark guided face Parsing (LaPa) dataset by JDAI-CV.

Upon training completion, model converted to TensorflowJS and integrated with React for realtime inference with client browser and webcam.

GPU-capable hardware recommended for optimal experience. 