# yield用法: 输出斐波那契数列
# v1 使用print导致函数可复用性差
def fab1(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        # 注意这里有个细节操作
        a, b = b, a + b 
        # a = b 
        # b = a + b 完全不同 
        n = n + 1
fab1(5)

# v2 返回List后能满足复用性的要求, 但是该函数运行在占的内存随着参数max的增大而增大
def fab2(max):
    n, a, b = 0, 0, 1
    L = []
    while n < max:
        L.append(b)
        a, b = b, a + b
        n = n + 1
    return L

for n in fab2(5):
    print(n)

# v3 利用 iterable 我们可以把 fab 函数改写为一个支持 iterable 的 class 
# Fab 类通过 next()不断返回数列的下一个数，内存占用始终为常数
class Fab3(object):
    def __init__(self, max):
        self.max = max
        self.n, self.a, self.b = 0, 0, 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.n < self.max:
            r = self.b
            self.a, self.b = self.b, self.a + self.b
            self.n = self.n + 1
            return r
        raise StopIteration()
for n in Fab3(5):
    print(n)

# v4 简单地讲，yield 的作用就是把一个函数变成一个 generator
# 带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个generator
# 调用 fab(5) 不会执行 fab 函数，而是返回一个 iterable 对象！
# 在 for 循环执行时，每次循环都会执行 fab 函数内部的代码，执行到 yield b 时
# fab 函数就返回一个迭代值，下次迭代时，代码从 yield b 的下一条语句继续执行
# 而函数的本地变量看起来和上次中断执行前是完全一样的，于是函数继续执行，直到再次遇到 yield。
def fab4(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1

for n in fab4(5):
    print(n)

# 一个带有 yield 的函数就是一个 generator，它和普通函数不同，
# 生成一个 generator 看起来像函数调用，但不会执行任何函数代码，
# 直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。
# 虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，
# 下次执行时从 yield 的下一个语句继续执行。看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，
# 每次中断都会通过 yield 返回当前的迭代值。