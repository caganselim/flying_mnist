import json
import numpy as np
import sys
import os
from PIL import Image
import math
import torch

class FlyingMNIST:

    def __init__(self, opts):

        # Save options
        self.opts = opts

        # Load full dataset
        self.mnist = None
        self.labels = None

        # Video related variables, will be reset for each video
        self.grayscale_digits = []
        self.colored_digits = []
        self.number_of_digits = None
        self.digit_sizes = None
        self.digit_labels = None
        self.colors = None
        self.coor_list = None
        self.xlim = None
        self.ylim = None
        self.veloc = None
        self.PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]

        self.frame_idx = 0
        self.vid_idx = 0

        self.create_dirs()
        self.load_dataset()

    def load_dataset(self):

        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)

            data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
            return data #/ np.float32(255)

        def load_labels(filename):

            if not os.path.exists(filename):
                download(filename)

            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)

            return data

        if self.opts.use_trn:

            self.mnist = load_mnist_images('train-images-idx3-ubyte.gz')
            self.labels = load_labels('train-labels-idx1-ubyte.gz')

        else:

            self.mnist = load_mnist_images('t10k-images-idx3-ubyte.gz')
            self.labels = load_labels('t10k-labels-idx1-ubyte.gz')

    def create_dirs(self):

        os.mkdir(self.opts.target_dir)

        for i in range(self.opts.num_videos):
            video_dir = os.path.join(self.opts.target_dir, f'{i:05d}')
            os.mkdir(video_dir)
            os.mkdir(os.path.join(video_dir, "flow"))
            os.mkdir(os.path.join(video_dir, "seg"))
            os.mkdir(os.path.join(video_dir, "vid"))

    def init_env(self, vid_idx = 0):

        self.vid_idx = vid_idx
        self.frame_idx = 0

        # Randomly select digit indices
        self.number_of_digits = np.random.randint(low=1, high=self.opts.max_digits + 1)

        # Generate random digit size
        digit_sizes = np.random.randint(self.opts.digit_size_min, self.opts.digit_size_max, self.number_of_digits)

        # Select digits
        digit_indices = []
        digit_labels = np.random.choice(np.array(self.opts.digits), self.number_of_digits)

        self.digit_labels = digit_labels

        for i in range(self.number_of_digits):

            # Returns a boolean array that is true for a specific digit, filter_digits[i]
            bools = np.isin(self.labels, int(digit_labels[i]))
            idxs = np.where(bools)[0]

            # Select a digit index randomly and save it.
            digit_idx = np.random.choice(idxs)
            digit_indices.append(digit_idx)

        # Generate tuples of (x,y) proper initial positions.
        # Each coordinate (x,y) defines upper-left corner of the digit. (xmin,ymin)
        self.xlim = self.opts.canv_width - self.opts.digit_size_max - 1
        self.ylim = self.opts.canv_height - self.opts.digit_size_max - 1
        self.coor_list = np.floor(np.asarray([(np.random.rand() * self.xlim, np.random.rand() * self.ylim) for _ in range(self.number_of_digits)]))

        # Velocity init
        direcs = np.pi * (np.random.rand(self.number_of_digits) * 2 - 1)
        speeds = np.random.randint(self.opts.max_speed, size=self.number_of_digits) + 10
        self.veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])

        # Select colors from a linear space
        color_basis = np.linspace(0, 1, 25)[4:]

        # Resize MNIST digits w.r.t sizes. Get size of each digit and interpolate it
        colored_digits = []
        grayscale_digits = []
        for i in range(self.number_of_digits):

            size = digit_sizes[i]
            idx = digit_indices[i]
            digit = self.mnist[idx].transpose(2,1,0)
            im = Image.fromarray(digit[:,:,0]).resize((size,size), Image.ANTIALIAS).convert("L")

            grayscale_digits.append(im)
            colored_digit = np.repeat(digit, repeats=3, axis=2)/255.

            if self.opts.use_coloring:
                color = np.random.choice(color_basis,3)
            else:
                color = np.ones((3))

            # Apply colors
            colored_digit[:, :, 0] *= color[0]
            colored_digit[:, :, 1] *= color[1]
            colored_digit[:, :, 2] *= color[2]
            colored_digit = (colored_digit*255.).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(colored_digit).resize((size,size), Image.ANTIALIAS).convert("RGB")
            colored_digits.append(im)

        self.colored_digits = colored_digits
        self.grayscale_digits = grayscale_digits

    def update_coords(self):

        # Get the next position by adding velocity
        next_coor_list = self.coor_list + self.veloc

        # Iterate over velocity and see if we hit the wall
        # If we do then change the  (change direction)
        for i, coord in enumerate(next_coor_list):

            # Calculate how many pixels can we move around a single image. => (x_lim, y_lim)
            xmin, ymin = coord[0], coord[1]

            # Check that if we hit the boundaries
            if xmin < 0 or xmin > self.xlim:
                self.veloc[i, 0] *= -1

            if ymin < 0 or ymin > self.ylim:
                self.veloc[i, 1] *= -1

            next_coor_list = self.coor_list + self.veloc

        # Update the coordinates
        self.coor_list = next_coor_list
        self.frame_idx += 1

    def generate_img(self):

        canvas = Image.new('RGB', (self.opts.canv_width,self.opts.canv_height))

        for i in range(self.number_of_digits):

            # Create a mask
            digit_bin = self.grayscale_digits[i]
            digit_mask = np.array(digit_bin)
            digit_mask[digit_mask < 40] = 0
            digit_mask[digit_mask > 0] = 255
            digit_mask = Image.fromarray(digit_mask).convert('L')

            # Prepare coords
            coor = np.floor(self.coor_list[i,:]).astype(int)
            coor = (coor[0], coor[1])

            # Paste it
            canvas.paste(self.colored_digits[i] , coor, digit_mask)

        return canvas


    def generate_seg(self):

        seg = Image.new('P', (self.opts.canv_width,self.opts.canv_height))
        seg.putpalette(self.PALETTE)

        for i in range(self.number_of_digits):

            # Create a mask
            digit_bin = self.grayscale_digits[i]
            digit_mask = np.array(digit_bin)
            digit_mask[digit_mask < 40] = 0
            digit_mask[digit_mask > 0] = 255
            digit_mask = Image.fromarray(digit_mask).convert('L')

            # Prepare coords
            coor = np.floor(self.coor_list[i,:]).astype(int)
            coor = (coor[0], coor[1])

            # Seg mask
            label_mask = np.array(digit_bin)
            label_mask[label_mask < 10] = 0
            label_mask[label_mask > 0] = i + 1
            instance = Image.fromarray(label_mask).convert('P')

            # Paste it
            seg.paste(instance , coor, digit_mask)

        return seg

    def generate_flow(self):

        flow = torch.zeros((1, 2, self.opts.canv_height, self.opts.canv_width), dtype=torch.float32)

        # Now we have the velocity and positions. Calculate flow:
        for i in range(self.number_of_digits):

            # Get image
            im = self.grayscale_digits[i]
            height, width = im.size

            im = torch.tensor(np.array(im))

            # Apply thresholding: convert to a mask
            im[im < 10] = 0
            im[im > 0] = 255

            # Get coordinates
            x = int(self.coor_list[i][0])
            y = int(self.coor_list[i][1])

            # Update flows. Perform masking for the accurate flow.

            # First, extract mask from the pre-existing flow.
            f_x = flow[:, 0:1, y:y + height, x:x + width]
            f_y = flow[: ,1:2, y:y + height, x:x + width]

            print(f"x: {x}, y: {y}, height: {height}, width: {width}")

            print(f_x.shape)
            print(f_y.shape)

            # Then apply AND operation to mask the existing flow.
            mask_x = torch.logical_and((f_x != 0), (im > 0))
            mask_y = torch.logical_and((f_y != 0), (im > 0))

            f_x[mask_x] = 0
            f_y[mask_y] = 0

            flow[:, 0:1, y:y + height, x:x + width] = f_x
            flow[: ,1:2, y:y + height, x:x + width] = f_y

            flow[:, 0:1, y:y + height, x:x + width] += self.veloc[i][0] * im # x direction
            flow[: ,1:2, y:y + height, x:x + width] += self.veloc[i][1] * im # y direction

        return flow

    def write_data(self):

        # Generate data
        img = self.generate_img()
        seg = self.generate_seg()
        flow = self.generate_flow()
        video_dir = os.path.join(self.opts.target_dir, f'{self.vid_idx:05d}')

        img.save(os.path.join(video_dir, "vid", f'{self.frame_idx:05d}.png'))
        seg.save(os.path.join(video_dir, "seg", f'{self.frame_idx:05d}.png'))
        torch.save(flow, os.path.join(video_dir, "flow", f'{self.frame_idx:05d}.pt'))

    def generate(self):

        print("Generating Flying MNIST dataset...")

        for vid_idx in range(self.opts.num_videos):

            print(f"Processing video: {vid_idx}/{self.opts.num_videos}")

            self.init_env(vid_idx)

            for frame_idx in range(self.opts.num_frames):
                self.write_data()
                self.update_coords()
