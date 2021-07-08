import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES

import json
import copy 

@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        origin = copy.deepcopy(data)

        for t in self.transforms:
            try : 
                data = t(data)
            except Exception as e:
                print(e)
                with open('log.txt','a') as f :
                    #json.dump(origin,f)
                    #f.write(str(origin['img_info']['id'])+'\n')
                    f.write(str(origin['img_info']['file_name'])+'\n')
                return None

            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
