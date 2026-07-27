
def eq_1(tid, reg):
    t0 = tid % 4
    t1 = (tid // 4) % 8
    t2 = tid // 32
    r0 = reg % 2
    r1 = (reg // 2) % 2
    r2 = reg // 4
    lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512
    row = lin % 64 
    col = lin // 64
    return row, col

def eq_2(tid, reg):
    chunk = reg // 32
    reg = reg % 32
    t0 = tid % 4
    t1 = (tid // 4) % 8
    t2 = tid // 32
    r0 = reg % 2
    r1 = (reg // 2) % 2
    r2 = reg // 4
    lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512
    row = lin % 64 
    col = chunk * 64 + lin // 64
    return row, col

for tid in range(128):
    for reg in range(128):
        tid = int(tid)
        reg = int(reg)
        o1 = eq_1(tid, reg)
        o2 = eq_2(tid, reg)
        assert o1 == o2


idx = []
for row in range(64):
    cols = list(range(8))
    idx.append(cols)
        

for tid in range(128):
    for reg in range(4):
        row, col = eq_1(tid, reg)
        idx[row][col] = -1

assert all([x == [-1] * 8 for x in idx])