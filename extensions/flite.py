# ==============================================================================
# Copyright (c) 2018-2019, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """


import subprocess
from extensions.phoneset.phoneset import Phoneset


class Flite:

    def __init__(self, binary_path, phoneset_path, args=["-ps"]):
        self.binary_path = binary_path
        self.args = args
        self._phone_set = Phoneset(phoneset_path)

    def command(self, arg):
        return [self.binary_path] + self.args + [arg, 'none']

    def convert_to_phoneme(self, text):
        command = self.command(text)
        result = subprocess.run(command, stdout=subprocess.PIPE, check=True)
        phone_txt = result.stdout.decode('utf-8', 'strict')
        phone_list = phone_txt.split(' ')
        phone_list = phone_list[:-1] if phone_list[-1] == '\n' else phone_list
        phone_ids = [self._phone_set.phone_to_id(p) for p in phone_list]
        return phone_ids, phone_txt

