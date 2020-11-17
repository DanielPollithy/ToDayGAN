import os.path, glob
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import code

class UnalignedDataset(BaseDataset):
    def __init__(self, opt, img_list=None):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)

        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

        if img_list:
            # sort the list according to the img_list
            only_name = lambda p: os.path.split(p)[-1]
            img_list_names = list(map(only_name, img_list))
            night_query_names = list(map(only_name, self.paths[1]))
            # check that all images are present
            assert set(night_query_names) == set(img_list_names)

            new_list = []
            new_prefix = self.paths[1][0].rsplit('/', maxsplit=1)[0]
            for i, img_name in enumerate(img_list_names):
                new_list.append(os.path.join(new_prefix, img_name))

            assert set(new_list) == set(self.paths[1])

            # Now the test images have the same sorting as the query list
            self.paths[1] = new_list

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def __getitem__(self, index):
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d,s in enumerate(self.sizes):
                    if index < s:
                        DA = d; break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # Choose two of our domains to perform a pass on
            DA, DB = random.sample(range(len(self.dirs)), 2)
            index_A = random.randint(0, self.sizes[DA] - 1)

        A_img, A_path = self.load_image(DA, index_A)
        bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        if self.opt.isTrain:
            index_B = random.randint(0, self.sizes[DB] - 1)
            B_img, _ = self.load_image(DB, index_B)
            bundle.update( {'B': B_img, 'DB': DB} )

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'UnalignedDataset'
