import tensorflow as tf


def close_crop(image, patch_size):

    image.set_shape([None, None, 3])
    width = 178
    height = 218
    new_width = 140
    new_height = 140

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2

    image = tf.expand_dims(image, axis=0)
    crops = tf.image.crop_to_bounding_box(image, top, left, bottom - top, right - left)

    resize = tf.image.resize_images(crops, [patch_size, patch_size])

    output = tf.squeeze(resize, axis=0)
    output.set_shape([patch_size, patch_size, 3])

    output = tf.to_float(output) / 255.

    return output


def image_file_inputs(file_patters, batch_size=32, patch_size=32):

    dataset = (tf.data.Dataset.list_files(file_patters)
               .map(tf.read_file)
               .map(tf.image.decode_image)
               .map(lambda x: close_crop(x, patch_size))
               .batch(batch_size))

    data_iterator = dataset.make_one_shot_iterator()

    images = data_iterator.get_next()

    return images, images
