#!/usr/bin/python3

import argparse
import face_recognition as fr
import os
import shutil
import pickle
import signal
import sys


class FaceOff:

    image_file_types = ('.jpg', '.jpeg', '.png', '.gif')

    def __init__(self, _options):
        self.source_directory = _options.source
        if not os.path.isdir(self.source_directory):
            print('ERROR: source directory must exist')
            sys.exit(1)

        if not any(filename.endswith(self.image_file_types) for filename in os.listdir(self.source_directory)):
            print('ERROR: source directory must contain image files')
            sys.exit(1)

        if _options.target:
            self.target_directory = _options.target
        else:
            parser.print_help()
            print('ERROR: target directory must be specified')
            sys.exit(1)
        if not os.path.isdir(self.target_directory):
            print('ERROR: target directory must exist')
            sys.exit(1)

        if os.path.exists('./face_encodings.pkl'):
            with open('face_encodings.pkl', 'rb') as input_file:
                self.processed_face_encodings = pickle.load(input_file)
        else:
            self.processed_face_encodings = []

        if os.path.exists('./face_directories.pkl'):
            with open('face_directories.pkl', 'rb') as input_file:
                self.processed_face_directories = pickle.load(input_file)
        else:
            self.processed_face_directories = []

        if self.source_directory[-1] != '/':
            self.source_directory += '/'
        if self.target_directory[-1] != '/':
            self.target_directory += '/'

        self.face_counter = len(self.processed_face_encodings)

    def run(self, _options):
        for file in os.listdir(self.source_directory):
            if file.endswith(self.image_file_types):
                try:
                    image = fr.load_image_file(self.source_directory + file)
                    face_locations = fr.face_locations(image, number_of_times_to_upsample=0, model='cnn')
                    face_encodings = fr.face_encodings(image, face_locations)
                    if len(face_encodings) == 0:
                        os.mkdir(self.target_directory + 'no_face_found')
                        shutil.copyfile(self.target_directory + file, self.target_directory + 'no_face_found/' + file)
                    for face_encoding in face_encodings:
                        matches = fr.compare_faces(self.processed_face_encodings, face_encoding)
                        if True in matches:
                            match_index = matches.index(True)
                            face_id = self.processed_face_directories[match_index]
                        else:
                            face_id = 'face%d' % self.face_counter
                            self.face_counter += 1
                            os.mkdir(self.target_directory + face_id)

                        self.processed_face_encodings.append(face_encoding)
                        self.processed_face_directories.append(face_id)

                        shutil.copyfile(self.target_directory + file, self.target_directory + face_id + '/' + file)

                except Exception as err:
                    print('ERROR: %s' % err)

        with open('face_encodings.pkl', 'wb') as output_file:
            pickle.dump(self.processed_face_encodings, output_file, pickle.HIGHEST_PROTOCOL)

        with open('face_directories.pkl', 'wb') as output_file:
            pickle.dump(self.processed_face_directories, output_file, pickle.HIGHEST_PROTOCOL)


def exit_gracefully(_signal, _frame):
    print('\rGoodbye!')
    exit()


if __name__ == '__main__':

    VERSION = '20180420_1510'
    desc = """
****************************************
 FaceOff - Quick and dirty image sorter
 Version %s
 Author: greatwhitehat (c) 2018
****************************************
""" % VERSION

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)
    parser.add_argument('--source', help='Source directory')
    parser.add_argument('--target', help='Target directory')
    options = parser.parse_args()

    if not (options.source and options.target):
        parser.print_help()
        print('ERROR: source and target directory must be specified')
        sys.exit(1)

    # Catch CTRL+C and exit in a nice way
    signal.signal(signal.SIGINT, exit_gracefully)

    face_off = FaceOff(options)
    face_off.run(options)
