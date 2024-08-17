class FibonacciCalculator:
    def __init__(self):
        self.memo = {}

    def calculate(self, n):
        if n <= 0:
            return []

        if n == 1:
            return [0]

        if n == 2:
            return [0, 1]

        if n in self.memo:
            return self.memo[n]

        fib_sequence = self.calculate(n - 1)
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        self.memo[n] = fib_sequence

        return fib_sequence