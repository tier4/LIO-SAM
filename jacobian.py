from sympy import diff, symbols, Matrix, sin, cos, pprint
from sympy.utilities.codegen import codegen

a, b, c = symbols("a,b,c")
x, y, z = symbols("x,y,z")

X = Matrix([[      1,       0,       0],
            [      0,  cos(x), -sin(x)],
            [      0,  sin(x),  cos(x)]])

Y = Matrix([[ cos(y),       0,  sin(y)],
            [      0,       1,       0],
            [-sin(y),       0,  cos(y)]])

Z = Matrix([[ cos(z), -sin(z),       0],
            [ sin(z),  cos(z),       0],
            [      0,       0,       1]])


def print_jacobian(M):
    pprint(diff(M, x))
    print("")
    pprint(diff(M, y))
    print("")
    pprint(diff(M, z))
    print("")


M = Z * Y * X
print_jacobian(M)

# codegen(("f", diff(M, x)), "C89", "jx", header=True, empty=True, to_files=True)
# codegen(("f", diff(M, y)), "C89", "jy", header=True, empty=True, to_files=True)
# codegen(("f", diff(M, z)), "C89", "jz", header=True, empty=True, to_files=True)
