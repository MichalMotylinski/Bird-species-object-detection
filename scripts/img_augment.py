import os
import argparse
from shutil import copyfile, rmtree
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import xml.etree.ElementTree
import random
import PIL


# Calculate new bounding box coordinates
def bnb_flip(attribute, bnb_max, bnb_min):
    xmin = float(attribute) - float(bnb_max)
    xmax = float(attribute) - float(bnb_min)
    return str(xmin), str(xmax)


# Create directory structure for augmented images
def clean_dir(output_dir):
    if os.path.exists(output_dir):
        rmtree(output_dir)
    img_dir = os.path.join(output_dir, "image")
    xml_dir = os.path.join(output_dir, "xml")
    os.makedirs(os.path.join(output_dir, "image"))
    os.mkdir(os.path.join(output_dir, "xml"))
    return output_dir, img_dir, xml_dir


# Copy original images with bounding boxes information
def copy_images(xml_input_dir, img_input_dir, img_dir, xml_dir):
    for file in os.listdir(xml_input_dir):
        # copy original bounding box coordinates
        copyfile(os.path.join(xml_input_dir, file), os.path.join(xml_dir, file))

        # copy original images
        image = file.split(".")[0] + ".jpg"
        copyfile(os.path.join(img_input_dir, image), os.path.join(img_dir, image))


def augment(img_dir, xml_dir):
    # Create augmented files
    for file in os.listdir(img_dir):
        if "jpg" in file:
            # Set variables to save information if augmentation method was applied
            rand_fliph = random.randint(0, 1)
            rand_flipv = random.randint(0, 1)
            # Make sure to apply at least one augmentation method
            if rand_fliph == 0 and rand_flipv == 0:
                rand_flipv = 1

            # Combine augmentation methods
            aug = iaa.Sequential([iaa.Fliplr(rand_fliph), iaa.Flipud(rand_flipv)])

            # Load image
            image = imageio.imread(os.path.join(img_dir, file))

            # Apply augmentation
            image_aug = aug(image=image)

            # Save augmented image
            filename = file.split(".")[0] + "_aug.jpg"
            im = PIL.Image.fromarray(image_aug)
            rgb_im = im.convert('RGB')
            rgb_im.save(os.path.join(img_dir, filename))

            # Now create new xml file with bounding box coordinates for augmented image
            # Copy original file and rename it
            filename = file.split(".")[0] + "_aug.xml"
            copyfile(os.path.join(xml_dir, file.split(".")[0] + ".xml"), os.path.join(xml_dir, filename))

            # Open image and get its attributes
            image = file.split(".")[0] + ".jpg"
            ims = PIL.Image.open(os.path.join(img_dir, image))
            width, height = ims.size

            # Open and read new xml file
            tree = xml.etree.ElementTree.parse(os.path.join(xml_dir, filename))
            root = tree.getroot()

            for child in root:
                if child.tag == "object":
                    for obj in child:
                        xmin = ""
                        xmax = ""
                        ymin = ""
                        ymax = ""
                        if obj.tag == "bndbox":
                            # Load current bounding box coordinates
                            for box in obj:
                                if box.tag == "xmin":
                                    xmin = box.text
                                if box.tag == "xmax":
                                    xmax = box.text
                                if box.tag == "ymin":
                                    ymin = box.text
                                if box.tag == "ymax":
                                    ymax = box.text

                            # Change values according to method applied
                            if rand_fliph == 1:
                                xmin, xmax = bnb_flip(width, xmax, xmin)
                            if rand_flipv == 1:
                                ymin, ymax = bnb_flip(height, ymax, ymin)

                            # Save new variables
                            for box in obj:
                                if box.tag == "xmin":
                                    box.text = xmin
                                if box.tag == "xmax":
                                    box.text = xmax
                                if box.tag == "ymin":
                                    box.text = ymin
                                if box.tag == "ymax":
                                    box.text = ymax
                # Change path and filename in the xml file pto match the name of the augmented image
                if child.tag == "path":
                    child.text = child.text.split(".")[0] + "_aug.jpg"
                if child.tag == "filename":
                    child.text = child.text.split(".")[0] + "_aug.jpg"
                    break
            # Save xml file
            tree.write(os.path.join(xml_dir, filename))


# Load arguments passed by user
def get_parser():
    parser = argparse.ArgumentParser(description="Apply augmentation to images")
    parser.add_argument("-i",
                        "--images",
                        type=str,
                        required=True,
                        help="Path to the folder where the input images are stored."
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
                        help="Path to the output folder where the augmentation dirs should be created."
                        )
    args = parser.parse_args()
    return args.images, args.xml, args.output


# Get user's input
img_input_dir, xml_input_dir, output_dir = get_parser()

# Create augment directories, remove any old entries
output_dir, img_dir, xml_dir = clean_dir(output_dir)

# Copy not augmented images
copy_images(xml_input_dir, img_input_dir, img_dir, xml_dir)

# Apply augmentation
augment(img_dir, xml_dir)
