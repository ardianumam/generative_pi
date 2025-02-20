import os
import numpy as np
from PIL import Image
from IPython.display import display

class PiDataset():
    def __init__(self, root="data"):            
        # load the x and y coordinates
        xs = np.load(os.path.join(root, 'pi_xs.npy'))  # Shape: (5000,)
        ys = np.load(os.path.join(root, 'pi_ys.npy'))  # Shape: (5000,)

        # load the image and get the RGB values
        image_array = np.array(Image.open('data/sparse_pi_colored.jpg'))
        rgb_values = image_array[xs, ys]  # Shape: (5000, 3)

        # combine x, y, and RGB values into a dataset
        self.data = np.column_stack((xs, ys, rgb_values)).astype(np.float32)  # Shape: (5000, 5)

        # store basic attributes
        self.root = root
        self.n_pts, self.n_feat = self.data.shape
        self.img_h, self.img_w, _ = image_array.shape
        
        # normalize the data into 0~1
        # self.data = self.normalize_data(self.data)

    def normalize_data(self, data):
        data[:,0] = data[:,0]/float(self.img_h)
        data[:,1] = data[:,1]/float(self.img_w)
        data[:,2:] = data[:,2:]/255.0
        return data
    
    def unnormalize_data(self, data):
        data[:,0] = data[:,0]*self.img_h
        data[:,1] = data[:,1]*self.img_w
        data[:,2:] = data[:,2:]*255
        data = data.astype(np.int32)
        return data

    def dump_data(self, data=None, dir_out=None, filename=None):
        assert data is not None, "data cannot be None!"
        assert data is not None, "filename cannot be None!"
        data = np.copy(data) # avoid in-place modification 

        if dir_out is None: dir_out = os.path.join("ouput")
        os.makedirs(dir_out, exist_ok=True)

        # create a blank image
        data_image = np.zeros((300, 300, 3), dtype=np.uint8)

        # transform the data back, from 0~1 into the original image size and color
        # data = self.unnormalize_data(data)

        # populate the image with the dataset points
        for x, y, r, g, b in data:
            if 0 <= x < 300 and 0 <= y < 300:
                data_image[int(x), int(y)] = [int(r), int(g), int(b)]

        # save the verification image
        data_image_pil = Image.fromarray(data_image)
        filepath = os.path.join(dir_out, f'{filename}.png')
        data_image_pil.save(filepath)

        print(f"The data is writen in: {filepath}")
    
    def show_data(self):
        # Create a blank image
        data_image = np.zeros((300, 300, 3), dtype=np.uint8)

        # Populate the image with the dataset points
        for x, y, r, g, b in self.data:
            data_image[int(x), int(y)] = [int(r), int(g), int(b)]

        # Save the verification image
        data_image_pil = Image.fromarray(data_image)
        display(data_image_pil)

    def  __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)