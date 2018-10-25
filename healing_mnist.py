import mnist
import numpy as np
import scipy.ndimage

def apply_square(img, square_size):
    img = np.array(img)
    img[:square_size, :square_size] = 255
    return img
    
def apply_noise(img, bit_flip_ratio):
    img = np.array(img)
    mask = np.random.random(size=(28,28)) < bit_flip_ratio
    img[mask] = 255 - img[mask]
    return img

def get_rotations(img, rotation_step, rotation_count):
    for i in range(rotation_count):
        yield img
        img = scipy.ndimage.rotate(img, rotation_step, reshape=False)

def heal_mnist(images, seq_len, rotation_step, square_count, square_size, noise_ratio):
    for img in images:
        squares_begin = np.random.randint(0, seq_len - square_count)
        squares_end = squares_begin + square_count

        rotations = []

        for idx, rotation in enumerate(get_rotations(img, rotation_step, seq_len)):
            if idx >= squares_begin and idx < squares_end:
                rotation = apply_square(rotation, square_size)
            rotations.append(apply_noise(rotation, noise_ratio))

        yield rotations

def train_images(seq_len=5, rotation_step=15, square_count=3, square_size=5, noise_ratio=0.15):
    return np.array(list(heal_mnist(mnist.train_images(), seq_len=seq_len, 
                                                     rotation_step=rotation_step,
                                                     square_count = square_count,
                                                     square_size = square_size,
                                                     noise_ratio = noise_ratio)))

def test_images(seq_len=5, rotation_step=15, square_count=3, square_size=5, noise_ratio=0.15):
    return np.array(list(heal_mnist(mnist.test_images(),  seq_len=seq_len, 
                                                     rotation_step=rotation_step,
                                                     square_count = square_count,
                                                     square_size = square_size,
                                                     noise_ratio = noise_ratio)))
