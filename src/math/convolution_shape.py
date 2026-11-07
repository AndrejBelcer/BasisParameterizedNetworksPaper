def GetConvolutionNumerator(x: int, k: int, d: int, s: int, p: int = 0) -> int:
    return d * (1 - k) + s + (x + 2 * p) - 1

def GetConvolutionShape(x: int, k: int, d: int, s: int, p: int = 0) -> int:
    return (d * (1 - k) + s + (x + 2 * p) - 1) // s

def GetRequiredP(x: int, k: int, d: int, s: int) -> int:
    # solve (d * (1 - k) + s + (x + 2 * p) - 1) / s == x/s for p
    # p = 1/2 (d (k - 1) - s + 1) and s!=0

    if(x % s != 0):
        raise ValueError("s doesn't divide x.")

    t = d * (k - 1) - s + 1
    if (t % 2 != 0):
        raise ValueError("Shape padding problem. Padding can't correct this.")
    p = t // 2

    y_1 = x // s
    y_2 = GetConvolutionShape(x, k, d, s, p)

    if (y_1 != y_2):
        raise ValueError("Shape padding problem. The calculation didn't produce the correct padding.")

    return p
