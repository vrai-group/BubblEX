# BubblEX
Point cloud explainability

A DGCNN model for 3d point clouds classification trained on the modelnet40 dataset was used. The test data include 2468 objects consisting of 1024 points each and belonging to 40 distinct classes. Model and data were sourced from https://github.com/AnTao97/dgcnn.pytorch

For each test object are performed in sequence:
- interpretation
- visualization

## Interpretation Module

### prediction and features extraction

prediction of the output and extraction of the features at the output of the layer "linear2" 

```
python predict.py
```

### tsne

"tsne" dimensionality reduction algorithm applied to the features at the output of the "linear2" layer

```
python 
```

## Visualization Module

### activation and gradient extraction

extraction of activation and gradient at the output of the layer "conv5" 

```
python actGradExtract.py
```

### gradcam

computation of the gradcam combining activation and gradient at the output of the "conv5" layer

```
python explain_gradcam.py
```

### bubble visualization

convert ply file to renderizable xml file

```
python ply2xmlRenderFile.py
```

use of mitsuba 0.5 to render. https://www.mitsuba-renderer.org/


If you want visualize also activation...

```
python explain_activation.py
```

