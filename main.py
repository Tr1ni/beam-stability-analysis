#python 3.10
#библиотеки 
import numpy as nmp
import sympy as smp
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

#символы
p, c1, c2, c3, c4, k, x = smp.symbols("p c1 c2 c3 c4 k x")

#константы
po = 8.0 * 10**3
E = 2.0 * 10**11
F = 7.0 * 10**(-3)
J = 4.0 * 10**(-6)
L = 2.5

#функции
#функция общего вида
def u(x):
    k = smp.root((po * F * p * p) / (E * J), 4)
    return c1 * smp.sin(k * x) + c2 * smp.cos(k * x) + c3 * smp.sinh(k * x) + c4 * smp.cosh(k * x)

#первая производная
def d_u(x1):
    return smp.diff(u(x), x).subs(x, x1)
                    
#построение матрицы
def Matrix() -> smp.Matrix:
    return smp.Matrix([[u(0).coeff(c1),u(0).coeff(c2),u(0).coeff(c3),u(0).coeff(c4)],
                [d_u(0).coeff(c1), d_u(0).coeff(c2), d_u(0).coeff(c3), d_u(0).coeff(c4)],
                [u(L).coeff(c1), u(L).coeff(c2), u(L).coeff(c3), u(L).coeff(c4)],
                [d_u(L).coeff(c1), d_u(L).coeff(c2), d_u(L).coeff(c3), d_u(L).coeff(c4)]])

#определитель матрицы
def det_M(): return smp.det(Matrix())

#Построение графика функции y(p) для нахождения pi_l
def plot_p():
    X = nmp.arange(0, 5001, 1)
    Y = smp.lambdify(p, det_M(), 'numpy')

    #для удобства здесь также рассчитаны точные значения pi_l
    global pi_l
    pi_l = list(fsolve(Y, (428, 1179, 2312)))

    plt.ylabel(r'$y(p)$', fontsize=14)
    plt.xlabel(r'$p$', fontsize=14)
    plt.xlim(0,5000)
    plt.ylim(-5,5)
    plt.grid(True)
    plt.savefig('figure_with_legend.png')
    plt.plot(X, Y(X))
    plt.show()

    
#Построение графика функции u(x)
def plot_ux(pi) -> None:
    #найдем значение k 
    k = pow((po * F * pi * pi) / (E * J), 1/4)

    #найдем коэффициенты
    global c1, c2, c3, c4
    c1 = 1
    u = c1 * smp.sin(k * x) + c2 * smp.cos(k * x) + c3 * smp.sinh(k * x) + c4 * smp.cosh(k * x)
    d_u = smp.diff(u, x)
    coef = list(smp.linsolve([u.subs(x, 0), d_u.subs(x, 0), d_u.subs(x, L)], [c2, c3, c4]))
    n_c1, n_c2, n_c3, n_c4 = c1, coef[0][0], coef[0][1], coef[0][2]

    #Объявим новую функцию u(x), полностью уйдя от символов и сделав функцию "численной", заменив тригонометрические функции из sympy
    #функциями из numpy
    def n_u(x):
        return n_c1 * nmp.sin(k * x) + n_c2 * nmp.cos(k * x) + n_c3 * nmp.sinh(k * x) + n_c4 * nmp.cosh(k * x)

    #построим график
    X = nmp.arange(0, L + 1, 0.1)
    plt.ylabel(r'$u(x)$', fontsize=14)
    plt.xlabel(r'$x$', fontsize=14)
    plt.xlim(0, L+1)
    plt.grid(True)
    plt.savefig('figure_with_legend.png')
    plt.plot(X, n_u(X))
    plt.show()

#Основная функция
def main():
    #Построим график y(p)
    plot_p() 

    #по графику p примерно равно:
    #p1 = 428
    #p2 = 1179
    #p3 = 2312

    #точные значения были высчитаны в функции plot_p(). Построим для получившихся значений три графика u(x)
    for i in range(0, 3):
        print('p%i' % (i + 1), ' - ' + str(pi_l[i]))
        plot_ux(pi_l[i])

if __name__ == "__main__":
    main()