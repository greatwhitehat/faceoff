#!/usr/bin/python3

import face_recognition as fr
import os
import shutil
import sys

print('Faceoff - Quick and dirty image sorter by face')
print('By greatwhitehat')

image_file_types = ('.jpg', '.jpeg', '.png', '.gif')

target_directory = sys.argv[1]

if target_directory[-1] != '/':
    target_directory += '/'
encoded_target_directory = os.fsencode(target_directory)

processed_face_encodings = []
processed_face_directories = []

face_counter = 0

for file in os.listdir(encoded_target_directory):
    filename = os.fsdecode(file)
    if filename.endswith(image_file_types):
        try:
            image = fr.load_image_file(target_directory + filename)
            face_locations = fr.face_locations(image, number_of_times_to_upsample=0, model='cnn')
            face_encodings = fr.face_encodings(image, face_locations)

            for face_encoding in face_encodings:
                matches = fr.compare_faces(processed_face_encodings, face_encoding)
                if True in matches:
                    match_index = matches.index(True)
                    face_id = processed_face_directories[match_index]
                else:
                    face_id = 'face%d' % face_counter
                    face_counter += 1
                    os.mkdir(target_directory + face_id)

                processed_face_encodings.append(face_encoding)
                processed_face_directories.append(face_id)

                shutil.copyfile(target_directory + filename, target_directory + face_id + '/' + filename)

        except Exception as err:
            print('Error: %s' % err)
            continue
        except KeyboardInterrupt:
            pass
    else:
        continue

