import os
from data import srdata
from data import common
import imageio
import torch.utils.data as data


class PPGScrub(srdata.SRData):
    def __init__(self, args, name='PPGScrub', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark
        # self.dir_hr = os.path.join(args.dir_demo, 'HR_images_X{}')
        self.dir_lr = os.path.join(args.dir_demo, 'images')
        # self.ext = ('.png', '.png')
        # self.dct = {'name': [], 'lr_path': [], 'hr_path': []}
        # cnt = 0
        self.filelist = []
        # import pdb; pdb.set_trace()
        for root, dirs, files in os.walk(self.dir_lr):
            # print("{} === {} === {}".format(files, dirs, root))
            for filename in files:
                # print(filename)
                # import pdb; pdb.set_trace()
                if filename.lower().find('.png') >=0 or filename.lower().find('.jp') >= 0:
                    self.filelist.append(os.path.join(root, filename))
                    # self.dct['name'].append(filename)
                    # self.dct['lr_path'].append(os.path.join(root, filename))
                    # self.dct['hr_path'].append(os.path.join(self.dir_hr, root.split('/')[-1]))
                    # print(self.dct['hr_path'][-1])
                    # cnt += 1
                    # import pdb; pdb.set_trace()
        self.filelist.sort()

    def __getitem__(self, idx):
        folder = self.filelist[idx].split('/')[-2]
        filename = folder + '/' + os.path.splitext(os.path.basename(self.filelist[idx]))[0]
        # print("working on {}".format(filename))
        lr = imageio.imread(self.filelist[idx])
        lr, = common.set_channel(lr, n_channels=self.args.n_colors)
        lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
        print("working on {}".format(filename))
        return lr_t, -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale