import time
import random
from Crypto.PublicKey import RSA

# 需求1: 加法
def add_test(times):
    a = random.getrandbits(2048)
    b = random.getrandbits(2048)
    start = time.time()
    for _ in range(times):
        _ = a + b
    return time.time() - start

# 需求1: 减法
def sub_test(times):
    a = random.getrandbits(2048)
    b = random.getrandbits(2048)
    start = time.time()
    for _ in range(times):
        _ = a - b
    return time.time() - start

# 需求1: 乘法
def mul_test(times):
    a = random.getrandbits(2048)
    b = random.getrandbits(2048)
    start = time.time()
    for _ in range(times):
        _ = a * b
    return time.time() - start

# 需求2: 整除
def div_test(times):
    a = random.getrandbits(2048)
    b = random.getrandbits(512)
    start = time.time()
    for _ in range(times):
        _ = a // b
    return time.time() - start

# 需求2: 取余
def mod_test(times):
    a = random.getrandbits(2048)
    b = random.getrandbits(512)
    start = time.time()
    for _ in range(times):
        _ = a % b
    return time.time() - start

# 需求3: 高次幂取余
def pow_mod_test(times):
    a = random.getrandbits(1024)
    b = random.getrandbits(1024)
    c = random.getrandbits(1024)
    start = time.time()
    for _ in range(times):
        _ = pow(a, b, c)
    return time.time() - start

# 需求4: RSA加密
def rsa_encryption_test(times):
    key = RSA.generate(1024, e=65537)
    x = random.getrandbits(16)
    start = time.time()
    for _ in range(times):
        _ = pow(x, key.e, key.n)
    return time.time() - start

# 需求4: RSA解密
def rsa_decryption_test(times):
    key = RSA.generate(1024, e=65537)
    x = random.getrandbits(16)
    ciphertext = pow(x, key.e, key.n)
    start = time.time()
    for _ in range(times):
        _ = pow(ciphertext, key.d, key.n)
    return time.time() - start

# 主程序
def main():
    test_cases = [100000, 1000000, 10000000, 100000000, 1000000000]
    for t in test_cases:
        print(f"Add test {t} times: {add_test(t):.6f}s")
        print(f"Sub test {t} times: {sub_test(t):.6f}s")
        print(f"Mul test {t} times: {mul_test(t):.6f}s")
        print(f"Div test {t} times: {div_test(t):.6f}s")
        print(f"Mod test {t} times: {mod_test(t):.6f}s")
    test_cases = [100, 1000, 10000, 100000, 1000000]
    for t in test_cases:
        print(f"Pow mod test {t} times: {pow_mod_test(t):.6f}s")
        print(f"RSA encryption test {t} times: {rsa_encryption_test(t):.6f}s")
        print(f"RSA decryption test {t} times: {rsa_decryption_test(t):.6f}s")

if __name__ == "__main__":
    main()
