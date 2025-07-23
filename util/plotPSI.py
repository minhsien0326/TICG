import numpy as np
import matplotlib.pyplot as plt

arr1 = np.genfromtxt('../lammps-grid/smolBox_NPT_XN4p69_BB.txt')
arr2 = np.genfromtxt('../lammps-grid/smolBox_NPT_XN4p69_BV.txt')

plt.rcParams['font.size'] = 14


plt.figure(figsize=(8, 6))
plt.plot(arr1[:, 0], arr1[:, 1], label = 'Binary blend', color='k')
plt.plot(arr2[:, 0], arr2[:, 1], label = 'Binary Vitrimer', color='gray')
plt.xlabel('Timestep')
plt.ylabel('Degree of phase separation')
plt.title(r'$\chi N \approx 4.7$, NPT ensemble') 
plt.ylim(0.7, 1)
plt.legend()
plt.tight_layout()
plt.show()  