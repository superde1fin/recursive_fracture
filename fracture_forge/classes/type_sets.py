import numpy as np
import ctypes as ct

class Holder:
    def __init__(self, lmp):
        self.__lmp = lmp
        self.__type_set_dict = dict()
        self.__current_typeset = 0
        self.__type_set_dict[0] = self.__lmp.gather_atoms("type", 0, 1)
        self.__ntype_sets = 1
        self.__nactive_typesets = 1

    def add_typeset(self, typeset):
        self.__type_set_dict[self.__ntype_sets] = list(map(int, typeset))
        self.__ntype_sets += 1
        self.__nactive_typesets += 1
        return self.__ntype_sets - 1

    def change_typeset(self, typeset_id):
        if typeset_id < 0 or typeset_id >= self.__ntype_sets or not self.__type_set_dict[typeset_id]:
            raise RuntimeError("Invalid type set id")
        data = (self.__lmp.get_natoms()*ct.c_int)(*self.__type_set_dict[typeset_id])
        self.__lmp.scatter_atoms("type", 0, 1, data)
        self.__current_typeset = typeset_id

    def delete_typeset(self, typeset_id):
        if typeset_id < 0 or typeset_id >= self.__ntype_sets or not self.__type_set_dict[typeset_id]:
            raise RuntimeError("Invalid type set id")
        if typeset_id == self.__current_typeset:
            raise RuntimeError("Cannot delete current type set")
        self.__type_set_dict[typeset_id]= []
        self.__nactive_typesets -= 1

    def get_current(self):
        return self.__current_typeset

    def get_ntype_sets(self):
        return self.__nactive_typesets
