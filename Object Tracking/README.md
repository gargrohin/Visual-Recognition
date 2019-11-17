# Object Tracking

## SORT

SORT is an algorithm to track moving objects in a video. It's precision and latency depends on the Image recognition algorithm it uses. For tracking, it relies on Kalman Filters prediction of the future prediction.

The original implementation by the authors of SORT use FasterRCNN. This implementation uses YOLOv3, a much faster algorithm that improves tracking speed. 


## Future Frame Prediction

Kalman Filters are a traditional and an efficient way of predicting the position of a moving object. But they only work for objects moving linearly and with a constant velocity. 

**Future frame prediction** using **Adverserial Video Generation** can be a viable alternative to Kalman Filters. We have implemented SORT but replaced the Kalman Filters with the outputs from a *GAN*. 

#### Please refer to the report for further information and detailed analysis of the results.


