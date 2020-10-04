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
# 64 => if calling 61 times(Dataset.count == "61"), change to Fuji
# 64 => if calling 54 times(Dataset.count == "115"), end of 1epoch
# 32 => if calling 123 times(Dataset.count == "123"), change to Fuji
# 32 => if calling 108 times(Dataset.count == "231"), end of 1epoch
_ONE_FILE_SIZE = 2000 # number of tensor in a picker file
_DATASET_PATH = "dataset/"
_SONY_long_path = _DATASET_PATH + "Sony/long/"
_SONY_short_path = _DATASET_PATH + "Sony/short/"
_FUJI_long_path = _DATASET_PATH + "Fuji/long/"
_FUJI_short_path = _DATASET_PATH + "Fuji/short/"
_32 = "32/"
_64 = "64/"
_END_OF_EPOCHS_64 = 115
_END_OF_EPOCHS_32 = 231


class Dataset:
    """
    Parameters
    ==========
    _switching : Sony => Fuji
    _count : How manu times the dataset was called('_count*_batchsize' is dataset tensor list index now)
    _size : Image Tensor size of Height and Width
    _batch_size : Batch Size
    _endover : Number of calls to end 1 epoch 
    """
    def __init__(self,size=64,batch_size=16):
        self._switching = True
        self._count = 114
        self._size = size
        self._batch_size = batch_size
        if self._size == 64:
            self._endover = int(_SONY_64/_ONE_FILE_SIZE) + int(_FUJI_64/_ONE_FILE_SIZE)
        else:
            self._endover = int(_SONY_32/self._ONE_FILE_SIZE) + int(_FUJI_32/_ONE_FILE_SIZE)
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
    @property
    def switching(self):
        return self._switching
    @property
    def endover(self):
        return self._endover
    @switch.setter
    def switch(self,x):
        self._switching = True
    @switching.setter
    def switching(self,x):
        if type(x) is not bool:
            raise ValueError("{} is not bool type.switching must have bool object".format(x))
        self._switching = x
    @count.setter
    def count(self,x):
        self._count = x
    @property
    def change_now_check(self):
        """
        condition
        =========
        count - 1 == 0
        count > sony check
        """
        if self.count - 1 == 0:
            return True
        a = True if self.count > self.sony else False
        b = self.switch
        return a and b
    def long_dataset(self):
        """
        if 0 return Sony long dataset and attention_maps else Fuji
        """
        if self.count == 1:
            print('~~~~~~~~~~~ "Sony" Long Data ~~~~~~~~~~~')
            return self.get_tensor_list(_SONY_long_path+"long_dic_{}.pickle".format(self._size))
        else:
            print('~~~~~~~~~~~ "Fuji" Long Data ~~~~~~~~~~~')
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
        with open(file_,"rb") as f:
            list_ = pickle.load(f)
        return list_

    @property
    def over_end_count(self):
        next_count = self.count + 1#check next count
        return True if next_count >= self._endover else False

    def check_end(self):
        """
        whether end of 1 epoch
        """
        return True if self.switching and self.over_end_count else False
    def count_reset(self):
        """"
        call at end of epoch, count => 0 and switching = False
        """
        self.count = 0
        self.switching = False

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
        if size == 64:
            if count > _END_OF_EPOCHS_64:
                raise RuntimeError("count over _END_OF_EPOCHS_64...")
        else:
            if count > _END_OF_EPOCHS_32:
                raise RuntimeError("count over _END_OF_EPOCHS_32...")
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
        if self.count > self.endover:#check count over endover
            raise RuntimeError('calling count over _END_OF_EPOCH_{}'.format(self.size))
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
    @classmethod
    def array_to_tensor(self,tensor_in_list):
        return torch.cat(tensor_in_list).reshape(len(tensor_in_list),*tensor_in_list[0].shape)
    @classmethod
    def search_long_data(self,long_dic,imageid_list):
        """
        Parameters
        ==========
        long_dic => dic(key:imageid_patch,value:image_tensor)
        imageid_list => list(imageid_path)
        """
        long_tensor = []
        for imageid in imageid_list:
            if long_dic[imageid] is None:
                raise RuntimeError("{} is not found in long_dic".format(imageid))
            long_tensor.append(long_dic[imageid])
        return self.array_to_tensor(long_tensor)

if __name__ == "__main__":
    print("Hello,{}".format(__file__))
