# faceoff
Quick and dirty image sorter by faces.

Credit goes to Adam Geitgey for the face_recognition python module.
https://github.com/ageitgey/face_recognition

As stated in the caveats over at https://github.com/ageitgey/face_recognition/blob/master/README.md


* The face recognition model is trained on adults and does not work very well on children. It tends to mix
  up children quite easy using the default comparison threshold of 0.6.
* Accuracy may vary between ethnic groups. Please see [this wiki page](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-face-recognition-works-well-with-european-individuals-but-overall-accuracy-is-lower-with-asian-individuals) for more details.
