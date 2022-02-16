import os
import os.path
import hashlib
import errno
from torchvision import transforms

dataset_stats = {
    'CIFAR10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                 'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                 'size' : 32},
    'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                 'size' : 32},   
    'ImageNet': {'mean': (0.485, 0.456, 0.406),
                 'std' : (0.229, 0.224, 0.225),
                 'size' : 224},      
    'TinyImageNet': {'mean': (0.4389, 0.4114, 0.3682),
                 'std' : (0.2402, 0.2350, 0.2268),
                 'size' : 64},  
                }

# transformations
def get_transform(dataset='cifar100', phase='test', aug=True, dgr=False):
    transform_list = []

    # get crop size
    crop_size = dataset_stats[dataset]['size']

    # get mean and std
    dset_mean = dataset_stats[dataset]['mean']
    dset_std = dataset_stats[dataset]['std']
    if dgr:
        if len(dset_mean) == 1:
            dset_mean = (0.0,)
            dset_std = (1.0,)
        else:
            dset_mean = (0.0,0.0,0.0)
            dset_std = (1.0,1.0,1.0)

    if phase == 'train' and aug:
        if dataset == 'ImageNet':
            transform_list.extend([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
        else:
            transform_list.extend([
                transforms.ColorJitter(brightness=63/255, contrast=0.8),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(crop_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
    else:
        if dataset == 'ImageNet':
            transform_list.extend([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
        else:
            transform_list.extend([
                    transforms.ToTensor(),
                    transforms.Normalize(dset_mean, dset_std),
                                    ])

    return transforms.Compose(transform_list)

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)