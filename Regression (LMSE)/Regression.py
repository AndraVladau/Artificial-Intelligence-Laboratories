class MyRegressor:
    def __init__(self) -> None:
        self.W = []
        self.X = []

    def get_parameters(self):
        return self.W

    def predict(self, x):
        l = []
        for i in range(len(x)):
            c = self.W[0][-1]
            for j in range(len(x[i])):
                c += float(self.W[j + 1][-1]) * x[i][j]
            l.append(c)
        return l

    def train(self, training_input_set, training_output_set):
        self.X = [[1] + x.copy() for x in training_input_set]
        XT = self.__transpose_matrix(self.X)
        P = self.__product(XT, self.X)
        P_Invers = self.__inversa(P)
        A = self.__product(P_Invers, XT)
        Y = [[el] for el in training_output_set.copy()]
        self.W = self.__product(A, Y)

    def __transpose_matrix(self, A):
        n, m = len(A), len(A[-1])
        XT = [[] for _ in range(m)]
        for i in range(n):
            for j in range(m):
                XT[j].append(A[i][j])
        return XT

    def __product(self, A, B):
        nr_lines = len(A)
        nr_cols = len(B[-1])
        prod = []
        for i in range(nr_lines):
            prod.append([])
            for j in range(nr_cols):
                prod[i].append(sum([A[i][k] * B[k][j] for k in range(len(A[i]))]))
        return prod

    def __elimLinCol(self, A, i: int, j: int):
        B = [A[k].copy() for k in range(len(A)) if k != i]
        for line in B:
            line.pop(j)
        return B

    def __determinant(self, A):
        if len(A) == 1:
            return A[-1][-1]
        return sum(((-1) ** j) * A[0][j] * self.__determinant(self.__elimLinCol(A, 0, j)) for j in range(len(A[0])))

    def __inversa(self, A):
        dA = self.__determinant(A)
        B = []
        nr_lines = len(A)
        nr_cols = len(A[-1])
        for i in range(nr_lines):
            B.append([])
            for j in range(nr_cols):
                B[i].append(((-1) ** (i + j)) * self.__determinant(self.__elimLinCol(A, i, j)) / dA)
        return B