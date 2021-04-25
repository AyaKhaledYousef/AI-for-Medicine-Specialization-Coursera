# Loss function for image segmentation:
- The soft **dice loss** is a popular loss function for segmentation models.
- The advantage of the soft **dice loss** is that it works well in the presence of imbalanced data.
- This is especially important in our task of brain tumor segmentation, when a very small fraction of the brain will be tumor regions.
- The soft **dice loss** will measure the error between our prediction map, P, and our ground truth map, G.
- The model optimizes this loss function to get better and better segmentations. 
