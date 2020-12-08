import os
from data import srdata


class PPG(srdata.SRData):
    def __init__(self, args, name='PPG', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(PPG, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, _ = super(PPG, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            filename = filename.replace('HR', 'LR')
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}{}'.format(
                        s, filename, self.ext[1]
                    )
                ))
        print('names_hr')
        [print(i.split('/')[-1]) for i in names_hr]
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')

        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.jpg', '.jpg')
