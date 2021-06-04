"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'pascal-voc', 'cub', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200'}
        assert(database in db_names)

        if database == 'cifar-10':
            return '/fs/vulcan-datasets/cifar-10-python'
        
        elif database == 'cifar-20':
            return '/fs/vulcan-datasets/cifar-100-python'

        elif database == 'stl-10':
            return '/vulcanscratch/mgwillia/stl10_binary'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return '/fs/vulcan-datasets/imagenet'

        elif database == 'pascal-voc':
            #return '/fs/vulcan-datasets/pascal_voc/'
            return '/cfarhomes/mgwillia/pascal-voc-modified'
        
        elif database == 'cub':
            return '/fs/vulcan-datasets/CUB/CUB_200_2011/'

        else:
            raise NotImplementedError
