import glob
import os
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(description="Clean XML files from white spaces and convert to single line string")
parser.add_argument("-i",
                    "--images",
                    type=str,
                    required=True,
                    help="Path to the folder where the image dataset is stored."
                    )
parser.add_argument("-x",
                    "--xml",
                    type=str,
                    required=True,
                    help="Path to the folder where the image bounding box information is stored."
                    )
parser.add_argument("-o",
                    "--output",
                    type=str,
                    required=True,
                    help="Path to the output folder where the train and test dirs should be created."
                    )

args = parser.parse_args()
clean_path = os.path.join(args.output, "cleaned")

if not os.path.exists(clean_path):
    os.makedirs(clean_path)
else:
    os.system(f"rm -rf {clean_path}")
    os.makedirs(f"{clean_path}")

images = os.listdir(args.images)

files=[]
for file in glob.glob(f"{args.xml}/*.xml"):
    with open(file, 'r') as f:
        with open(os.path.join(clean_path, os.path.basename(file)), 'w') as f1:
            for line in f:
                f1.write(line.rstrip().replace(" ", ""))
    filename = file.rsplit("/", 1)[1].split(".")[0]
    files.append(filename)

    for f2 in images:
        if filename == f2.split(".")[0]:
            copyfile(os.path.join(args.images, f2), os.path.join(clean_path, f2))
            images.remove(f2)
            break
