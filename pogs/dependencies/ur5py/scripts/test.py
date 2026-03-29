from ur5py.ur5 import UR5Robot
import numpy as np

if __name__=='__main__':
    ur = UR5Robot()
    input('enter to enter freedrive')
    ur.start_freedrive()
    input('enter tro end freedrive')
    ur.stop_freedrive()