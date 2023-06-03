mn = 9999999
mn_i = 0

mx = -9999999
mx_i = 0

# L=2
for i in range(1, 100):
    c = 10*i + (100-i)*i
    if c > mx:
        mx = c
        mx_i = i
    if c < mn:
        mn = c
        mn_i = i

print(mx_i, mx)
print(mn_i, mn)

mn = 9999999
mn_i = 0
mn_j = 0

mx = -9999999
mx_i = 0
mx_j = 0

# L=3
for i in range(1, 100):
    for j in range(1, 100-i):
        c = 10*i + i*j + j*(100-i-j)
        if c > mx:
            mx = c
            mx_i = i
            mx_j = j
        if c < mn:
            mn = c
            mn_i = i
            mn_j = j

print(mx_i, mx_j, mx)
print(mn_i, mn_j, mn)