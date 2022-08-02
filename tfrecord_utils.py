import tensorflow as tf
from imageio import imread
import sys


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_function(image_paths, labels, out_path):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.

    print("Converting: " + out_path)

    # Number of images. Used when printing the progress.
    num_images = len(image_paths)

    # Open a TFRecordWriter for the output-file.
    with tf.io.TFRecordWriter(out_path) as writer:

        # Iterate over all the image-paths and class-labels.
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            # Print the percentage-progress.
            print_progress(count=i, total=num_images - 1)

            # Load the image-file using matplotlib's imread function.
            img = imread(path)

            # Convert the image to raw bytes.
            img_bytes = img.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = {"image": wrap_bytes(img_bytes), "label": wrap_int64(label)}

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


def parse_function(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(
        serialized=serialized, features=features
    )

    # Get the image as raw bytes.
    image_raw = parsed_example["image"]

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.io.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # The serialized image is reshaped to the original shape.
    image = tf.reshape(image, [32, 32, 3])

    # The image is scaled between 0 and 1.
    image /= 255

    # Get the label associated with the image.
    label = parsed_example["label"]

    # The image and label are now correct TensorFlow types.
    return image, label


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()
