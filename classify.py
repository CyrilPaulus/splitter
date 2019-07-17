import os
from nudenet import NudeClassifier
from nudenet import NudeDetector
from tqdm import tqdm
from shutil import copyfile
# classifier = NudeClassifier('models/classifier_model')
detector = NudeDetector('models/detector_model')

# print(classifier.classify(
#     'splitted/_momoiroiro_Cute_or_sexy__😆_c4h2ar_2.jpg'))


def is_naked(features):
    naked_label = ['F_BREAST', 'F_GENITALIA', 'M_GENITALIA', 'M_BREAST']
    for feature in features:
        if feature['label'] in naked_label:
            return True

    return False


files = os.listdir('splitted')
for filename in tqdm(files):
    path = os.path.join('splitted', filename)
    out_dir = None
    if is_naked(detector.detect(path)):
        out_dir = 'naked'
    else:
        out_dir = 'not_naked'
    out_path = os.path.join(out_dir, filename)
    copyfile(path, out_path)
