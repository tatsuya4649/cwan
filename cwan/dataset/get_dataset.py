import torch
import glob
import pickle

"""
dataset tensor in one picker file  have 2000 tensor.
"""
#total training calling 346 times by every epoch
#346 * 2000 => 692000 Tensor
_SONY_64 = 122000 # calling 61 times
_SONY_32 = 246000 # calling 123 times
_FUJI_64 = 108000 # calling 54 times
_FUJI_32 = 216000 # calling 108 times
_ONE_FILE_SIZE = 2000 # number of tensor in a picker file
_SONY_long_path = "Sony/long/"
_SONY_short_path = "Sony/short/"
_FUJI_long_path = "Fuji/long/"
_FUJI_short_path = "Fuji/short/"
_32 = "32/"
_64 = "64/"

class Dataset:
    """
    Parameters
    ==========
    _switching : Sony => Fuji
    _count : How manu times the dataset was called('_count*_batchsize' is dataset tensor list index now)
    _size : Image Tensor size of Height and Width
    _batch_size : Batch Size
    """
    def __init__(self,size=64,batch_size=16):
        self._switching = False
        self._count = 0
        self._size = size
        self._batch_size = batch_size
        if self._size == 64:
            self._endover = int(_SONY_64/self._batch_size) + int(_FUJI_64/self._batch_size)
        else:
            self._endover = int(_SONY_32/self._batch_size) + int(_FUJI_32/self._batch_size)
        self.sony = _SONY_64 / _ONE_FILE_SIZE if self._size==64 else _SONY_32 / _ONE_FILE_SIZE
        self.fuji = _FUJI_64 / _ONE_FILE_SIZE if self._size==64 else _FUJI_32 / _ONE_FILE_SIZE
        self.path_32_64 = _64 if self._size == 64 else _32
    def plus(self):
        self.count += 1
    @property
    def count(self):
        return self._count
    @property
    def size(self):
        return self._size
    @property
    def switch(self):
        return not self._switching
    @switch.setter
    def switch(self,x):
        self._switching = True
    @count.setter
    def count(self,x):
        self._count = x
    @property
    def change_now_check(self):
        """
        count > sony check
        """
        a = True if self.count > self.sony else False
        b = self.switch
        return a and b
    def long_dataset(self):
        """
        if 0 return Sony long dataset and attention_maps else Fuji
        """
        if self.count == 1:
            return self.get_tensor_list(_SONY_long_path+"long_dic_{}.pickle".format(self._size))
        else:
            self.switch = True
            return self.get_tensor_list(_FUJI_long_path+"flong_dic_{}.pickle".format(self._size))
    @classmethod
    def get_tensor_list(self,path):
        """
        get tensor list from path
        ex.short_imageid_list,short_list
        """
        files = glob.glob(path)
        if len(files) == 0:
            raise RuntimeError("There is not '{}' ".format(path))
        file_ = files[0]
        with oepn(file_,"rb") as f:
            list_ = pickle.load(f)
        return list_

    @property
    def over_end_count(self):
        return True if self.count >= self._endover else False

    def check_end(self):
        """
        whether end of 1 epoch
        """
        return True if self.switch and self.over_end_count else False

    @classmethod
    def change_now(self,count):
        """
        count > sony check
        """
        return True if count > self.sony else False

    @classmethod
    def dataset_tensor(self,count,batch_size=16,size=64):
        """
        return short image_list and short imageid_list from dataset
        """
        sony = _SONY_64 / _ONE_FILE_SIZE if size==64 else _SONY_32 / _ONE_FILE_SIZE
        fuji = _FUJI_64 / _ONE_FILE_SIZE if size==64 else _FUJI_32 / _ONE_FILE_SIZE
        path_32_64 = _64 if size == 64 else _32
        if count < sony:
            file_name_number = _ONE_FILE_SIZE * count
            #path
            short_imageid_list_path = _SONY_short_path + path_32_64  + "imageid/" + "short_imageid_list_{}_{}.pickle".format(size,file_name_number)
            short_list_path = _SONY_short_path + path_32_64 + "image/" + "short_list_{}_{}.pickle".format(size,file_name_number)
            print(short_imageid_list_path)
            print(short_list_path)
        else:
            file_name_number = int(_ONE_FILE_SIZE * (count - sony))
            #path
            short_imageid_list_path = _FUJI_short_path + path_32_64 + "imageid/" + "fshort_imageid_list_{}_{}.pickle".format(size,file_name_number)
            short_list_path = _FUJI_short_path + path_32_64 + "image/" + "fshort_list_{}_{}.pickle".format(size,file_name_number)
            print(short_imageid_list_path)
            print(short_list_path)
        short_imageid_list = Dataset.get_tensor_list(short_imageid_list_path)
        short_list = Dataset.get_tensor_list(short_list_path)
        return short_imageid_list,short_list
    
    def dataset_tensor(self):
        """
        return short image_list and short imageid_list from dataset
        """
        if self.count < self.sony:
            file_name_number = _ONE_FILE_SIZE * self.count
            #path
            short_imageid_list_path = _SONY_short_path + self.path_32_64 + "imageid/" + "short_imageid_list_{}_{}.pickle".format(self.size,file_name_number)
            short_list_path = _SONY_short_path + self.path_32_64 + "image/" + "short_list_{}_{}.pickle".format(self.size,file_name_number)
            print(short_imageid_list_path)
            print(short_list_path)
        else:
            file_name_number = int(_ONE_FILE_SIZE * (self.count - self.sony))
            #path
            short_imageid_list_path = _FUJI_short_path + self.path_32_64 + "imageid/" + "fshort_imageid_list_{}_{}.pickle".format(self.size,file_name_number)
            short_list_path = _FUJI_short_path + self.path_32_64 + "image/" + "fshort_list_{}_{}.pickle".format(self.size,file_name_number)
            print(short_imageid_list_path)
            print(short_list_path)
        short_imageid_list = Dataset.get_tensor_list(short_imageid_list_path)
        short_list = Dataset.get_tensor_list(short_list_path)
        return short_imageid_list,short_list

if __name__ == "__main__":
    print("Hello,{}".format(__file__))
