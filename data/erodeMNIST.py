import socket
import numpy as np
from torchvision import datasets, transforms
import cv2 as cv


# from: https://github.com/edenton/svg/blob/master/data/moving_mnist.py

class ErodingMNIST(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=True):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False  # multi threaded loading
        self.channels = 1

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                     dtype=np.float32)


        idx = np.random.randint(self.N)
        digit, _ = self.data[idx]
        digit_eroded = digit.numpy().squeeze()/2
        aff = cv.warpAffine(digit_eroded, np.array([[1, 0, 2], [0, 1, 1]], dtype=float), (digit_eroded.shape[1], digit_eroded.shape[0]))
        kernel = np.array([[0, 0.3, 0], [0.3, 0.2, 0.3], [0, 0.3, 0]])
        erod = cv.filter2D(digit_eroded, kernel=kernel*2, ddepth=-1)
        ove = erod+aff
        ove[ove>1] = 1.
        cv.imwrite("digit_eroded.png", digit_eroded*255)
        cv.waitKey(0)
        cv.imwrite("aff.png", aff*255)
        cv.waitKey(0)
        cv.imwrite("erod.png", erod*255)
        cv.waitKey(0)
        cv.imwrite("ove.png", ove*255)
        cv.waitKey(0)
        import sys
        sys.exit()





        sx = np.random.randint(image_size - digit_size)
        sy = np.random.randint(image_size - digit_size)
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(-2, 3)

        for t in range(self.seq_len):
            if sy < 0:
                sy = 0
                if self.deterministic:
                    dy = -dy
                else:
                    dy = np.random.randint(1, 5)
                    dx = np.random.randint(-4, 5)
            elif sy >= image_size - 32:
                sy = image_size - 32 - 1
                if self.deterministic:
                    dy = -dy
                else:
                    dy = np.random.randint(-4, 0)
                    dx = np.random.randint(-4, 5)

            if sx < 0:
                sx = 0
                if self.deterministic:
                    dx = -dx
                else:
                    dx = np.random.randint(1, 5)
                    dy = np.random.randint(-4, 5)
            elif sx >= image_size - 32:
                sx = image_size - 32 - 1
                if self.deterministic:
                    dx = -dx
                else:
                    dx = np.random.randint(-4, 0)
                    dy = np.random.randint(-4, 5)
            
            digit_eroded = cv.filter2D(digit_eroded, kernel=kernel, ddepth=-1)
            x[t, sy:sy + 32, sx:sx + 32, 0] += digit_eroded
            if t>0:
                x[t, :, 0] += x[t-1, :, 0]
            sy += dy
            sx += dx

        x[x > 1] = 1.
        return x


