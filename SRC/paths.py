import os

here = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
root = os.path.dirname(here)

MODELDIR = os.path.join(root, "models")
DICTIONARYPATH = os.path.join(root, "dictionary.pkl")
TEXTDIR = os.path.join(root, "data/coco/vendrov/data/coco/")
IMGDIR = os.path.join(root, "data/coco/vendrov/data/coco/images/10crop")
CSVDIR = os.path.join(root, "CSV")
WEIGHTDIR = os.path.join(root, "weights")
