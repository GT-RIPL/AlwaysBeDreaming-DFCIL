from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch.utils.data as data
from .utils import download_url, check_integrity

"""
This file heavily adapts data-loading from Global Distillation
git url: https://github.com/kibok90/iccv2019-inc
@inproceedings{lee2019overcoming,
    title={Overcoming catastrophic forgetting with unlabeled data in the wild},
    author={Lee, Kibok and Lee, Kimin and Shin, Jinwoo and Lee, Honglak},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={312--321},
    year={2019}
}
"""

# for BiC dataloading
VAL_HOLD = 0.1
class iDataset(data.Dataset):
    
    def __init__(self, root,
                train=True, transform=None ,download_flag=False,
                tasks=None, seed=-1, validation=False, kfolds=5):

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag
        self.ic_dict = {}
        self.ic = False
        self.dw = True

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:
            
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

            # sample
            num_data_per_fold = int(len(self.targets) / kfolds)
            start = 0
            stop = num_data_per_fold
            locs_train = []
            locs_val = []
            for f in range(kfolds):
                if self.seed == f:
                    locs_val.extend(np.arange(start,stop))
                else:
                    locs_train.extend(np.arange(start,stop))
                start += num_data_per_fold
                stop += num_data_per_fold

            # train set
            if self.train:
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_train], task).nonzero()[0]
                    self.archive.append((self.data[locs_train][locs].copy(), self.targets[locs_train][locs].copy()))

            # val set
            else:
                self.archive = []
                for task in self.tasks:
                    locs = np.isin(self.targets[locs_val], task).nonzero()[0]
                    self.archive.append((self.data[locs_val][locs].copy(), self.targets[locs_val][locs].copy()))

        # else
        else:
            self.archive = []
            for task in self.tasks:
                locs = np.isin(self.targets, task).nonzero()[0]
                self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        if self.train:
            self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))


    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            if simple:
                img = self.simple_transform(img)
            else:
                img = self.transform(img)

        return img, self.class_mapping[target], self.t

    # dataset loading for BiC
    def load_bic_dataset(self, post = False):

        if post:
            self.data, self.targets = self.data_a, self.targets_a   

        else:

            # get number of holdout
            len_coreset = len(self.coreset[0])
            self.coreset_idx_change = int(VAL_HOLD * len_coreset)

            # get number of holdout for training data (balanced)
            num_class_past = 0
            for i_ in range(self.t):
                num_class_past += len(self.tasks[i_])
            k_per_class = int(self.coreset_idx_change / num_class_past)
            
            num_k_hold = [0 for i_ in range(1000)]
            idx_a = []
            idx_b = []
            for i_ in range(len(self.data)):

                k = self.targets[i_]
                if num_k_hold[k] < k_per_class:
                    idx_a.append(i_)
                else:
                    idx_b.append(i_)
                num_k_hold[k] += 1
            
            self.data_a, self.targets_a = self.data[idx_a], self.targets[idx_a]
            self.data, self.targets = self.data[idx_b], self.targets[idx_b]

    # append coreset for BiC and EtE
    def append_coreset_ic(self, post = False):
        if post:
            self.data = np.concatenate([self.data, self.coreset[0][self.coreset_sample_a_idx]], axis=0)
            self.targets = np.concatenate([self.targets, self.coreset[1][self.coreset_sample_a_idx]], axis=0)

        else:

            # get number of holdout for training data (balanced)
            num_class_past = 0
            for i_ in range(self.t):
                num_class_past += len(self.tasks[i_])
            k_per_class = int(self.coreset_idx_change / num_class_past)

            num_k_hold = [0 for i_ in range(1000)]
            idx_a = []
            idx_b = []
            for i_ in range(len(self.coreset[0])):

                k = self.coreset[1][i_]
                if num_k_hold[k] < k_per_class:
                    idx_a.append(i_)
                else:
                    idx_b.append(i_)
                num_k_hold[k] += 1

            self.coreset_sample_a_idx = idx_a
            self.data = np.concatenate([self.data, self.coreset[0][idx_b]], axis=0)
            self.targets = np.concatenate([self.targets, self.coreset[1][idx_b]], axis=0)

    # update coreset for BiC and EtE
    def update_coreset_ic(self, coreset_size, seen, teacher):
        self.ic = True
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)

            if not (k in self.ic_dict):

                # get numpy array of all feature embeddings
                feat_emb = []
                for loc in locs:

                    # get data to gpu
                    x, y, t = self.__getitem__(loc, simple=True)
                    x = x.cuda()
                    x = x[None,:,:,:]

                    # get feat embedding
                    z = teacher.generate_scores_pen(x)
                    feat_emb.append(z.detach().cpu().tolist())

                feat_emb = np.asarray(feat_emb)

                # calculate mean
                k_mean = np.mean(feat_emb, axis = 0)
                k_dist = feat_emb - k_mean[:]
                k_dist = np.squeeze(k_dist)
                k_dist = np.linalg.norm(k_dist, axis = 1)

                locs_chosen = []
                locs_k_array = np.arange(len(feat_emb))
                feat_emb_cp = np.copy(feat_emb)
                for k_ in range(num_data_k):

                    if len(locs_k_array) == 0:
                        pass
                    elif len(locs_k_array) == 1:
                        # append to save array
                        p_idx = 0
                        locs_chosen.append(locs_k_array[p_idx])

                        # remove from calculate array
                        locs_k_array = np.delete(locs_k_array, p_idx, axis = 0)
                        feat_emb_cp = np.delete(feat_emb_cp, p_idx, axis = 0)
                    else:

                        # get idx of closest to mean
                        chosen_feat = feat_emb[locs_chosen]
                        new_sum = np.sum(chosen_feat, axis = 0)
                        term_b = (feat_emb_cp + new_sum) / (len(locs_chosen) + 1)
                        term_b = np.squeeze(term_b)
                        k_dist_loop = k_mean - term_b
                        k_dist_loop = np.squeeze(k_dist_loop)
                        k_dist_loop = np.linalg.norm(k_dist_loop, axis = 1)
                        p_idx = np.argmin(k_dist_loop)
                        
                        # append to save array
                        locs_chosen.append(locs_k_array[p_idx])

                        # remove from calculate array
                        locs_k_array = np.delete(locs_k_array, p_idx, axis = 0)
                        feat_emb_cp = np.delete(feat_emb_cp, p_idx, axis = 0)

                # partition data
                locs_chosen = locs[locs_chosen]
                self.ic_dict[k] = [[self.data[loc] for loc in locs_chosen], [self.targets[loc] for loc in locs_chosen]]

            data.append(self.ic_dict[k][0][:num_data_k])
            targets.append(self.ic_dict[k][1][:num_data_k])
            
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))

    # update coreset for EtE
    def update_coreset_ete(self, coreset_size, seen, teacher):
        self.ic = True
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)

            if not (k in self.ic_dict):

                # get numpy array of all feature embeddings
                feat_emb = []
                for loc in locs:

                    # get data to gpu
                    x, y, t = self.__getitem__(loc, simple=True)
                    x = x.cuda()
                    x = x[None,:,:,:]

                    # get feat embedding
                    z = teacher.generate_scores_pen(x)
                    feat_emb.append(z.detach().cpu().tolist())

                feat_emb = np.asarray(feat_emb)

                # calculate mean
                k_mean = np.mean(feat_emb, axis = 0)
                k_dist = feat_emb - k_mean[:]
                k_dist = np.squeeze(k_dist)
                k_dist = np.linalg.norm(k_dist, axis = 1)

                locs_chosen = []
                locs_k_array = np.arange(len(feat_emb))
                feat_emb_cp = np.copy(feat_emb)
                for k_ in range(num_data_k):

                    if len(locs_k_array) == 0:
                        pass
                    elif len(locs_k_array) == 1:
                        # append to save array
                        p_idx = 0
                        locs_chosen.append(locs_k_array[p_idx])

                        # remove from calculate array
                        locs_k_array = np.delete(locs_k_array, p_idx, axis = 0)
                        feat_emb_cp = np.delete(feat_emb_cp, p_idx, axis = 0)
                    else:

                        # get idx of closest to mean
                        k_dist_loop = k_mean - feat_emb_cp
                        k_dist_loop = np.squeeze(k_dist_loop)
                        k_dist_loop = np.linalg.norm(k_dist_loop, axis = 1)
                        p_idx = np.argmin(k_dist_loop)
                        
                        # append to save array
                        locs_chosen.append(locs_k_array[p_idx])

                        # remove from calculate array
                        locs_k_array = np.delete(locs_k_array, p_idx, axis = 0)
                        feat_emb_cp = np.delete(feat_emb_cp, p_idx, axis = 0)

                # partition data
                locs_chosen = locs[locs_chosen]
                self.ic_dict[k] = [[self.data[loc] for loc in locs_chosen], [self.targets[loc] for loc in locs_chosen]]

            data.append(self.ic_dict[k][0][:num_data_k])
            targets.append(self.ic_dict[k][1][:num_data_k])
            
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))

    def load_dataset(self, t, train=True):
        
        if train:
            self.data, self.targets = self.archive[t] 
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

        print(np.unique(self.targets))

    # naive coreset appending
    def append_coreset(self, only=False, interp=False):
        len_core = len(self.coreset[0])
        if self.train and (len_core > 0):
            if only:
                self.data, self.targets = self.coreset
            else:
                len_data = len(self.data)
                sample_ind = np.random.choice(len_core, len_data)
                if self.ic:
                    self.data = np.concatenate([self.data, self.coreset[0]], axis=0)
                    self.targets = np.concatenate([self.targets, self.coreset[1]], axis=0)
                else:
                    self.data = np.concatenate([self.data, self.coreset[0][sample_ind]], axis=0)
                    self.targets = np.concatenate([self.targets, self.coreset[1][sample_ind]], axis=0)

    # naive coreset update
    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        
        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed*10000+self.t)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def load(self):
        pass

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class iCIFAR10(iDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iDataset Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    im_size=32
    nch=3

    def load(self):

        # download dataset
        if self.download_flag:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or self.validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iCIFAR10 Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    im_size=32
    nch=3




