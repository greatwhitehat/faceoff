# faceoff
Quick and dirty image sorter by faces.
The script will attempt to sort images by face and create a sub-directory with the face ID as directory name.
The original images will be copied to the new sub-directory to make sure data is not lost.

Make sure to run the pre_req_install.sh before running the script.

## Usage
python3 faceoff.py source/directory/ target/directory/

## Credits
Credit goes to Adam Geitgey for the face_recognition python module.
https://github.com/ageitgey/face_recognition

## Caveats
The script does not perform recursion within directories.
The target directory should only contain image files, no sub-directories.
I have not tested with images with several faces, the expected behaviour should be that the same image will end up in two directories.

As stated in the caveats over at https://github.com/ageitgey/face_recognition/blob/master/README.md

* The face recognition model is trained on adults and does not work very well on children. It tends to mix
  up children quite easy using the default comparison threshold of 0.6.
* Accuracy may vary between ethnic groups. Please see [this wiki page](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-face-recognition-works-well-with-european-individuals-but-overall-accuracy-is-lower-with-asian-individuals) for more details.
