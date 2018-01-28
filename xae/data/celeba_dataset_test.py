import tensorflow as tf
import numpy as np
from PIL import Image

from xae.data.celeba_dataset import image_file_inputs


images = image_file_inputs('/Users/jaonary/Data/celebA/img_align_celeba/*.jpg')


with tf.Session() as sess:


    for i in range(10):
        img = sess.run(images)
        pil_img = Image.fromarray(np.uint8(img[0])).convert('RGB')
        with tf.gfile.Open("image_{}.png".format(i), 'w') as fid:
            pil_img.save(fid, 'PNG')
