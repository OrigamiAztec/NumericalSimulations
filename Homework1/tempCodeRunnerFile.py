    sum = 0
        inc = (b - a) / n
        print(inc)
        for k in range(n + 1):
            x = a + (k * inc)
            summand = input_array[k]
            if (k != 0) and (k != n):
            summand *= (2 + (2 * (k % 2)))
            sum += summand
        return ((b - a) / (3 * n)) * sum
