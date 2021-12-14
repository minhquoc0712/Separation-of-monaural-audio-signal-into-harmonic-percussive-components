import numpy as np

a = [[1, 2, 3, 4, 5], [6, 7, 8, 9 , 10], [11, 12, 13, 14, 15]]

b = a.copy()

b = np.roll(b, len(b[0]))

print(a)
print(b)

e = 1
c = d = e

d = d + 1

print(c, d, e)

print(np.exp(1))

f = 1 * 1j

a_complex = a.copy()

for i in range(len(a)):
    for j in range(len(a[i])):
        a_complex[i][j] *= 1j

print(a_complex)

s = 'afjksm.wav'

print(s[:-4])