def ex1_1(hrs, mins):
    if hrs == 2576 and mins == 50:
        print('Pass!')
    else:
        print('Fail: Value Error')
    return
def ex1_2(accounts):
    if accounts == 'J0RB8BMRQU,LEKUUWH7QG,ED61NHVAKH,BLNJKE6MOT,WX79DJ3GBX,89RNWHB29T,DBSDKRY0M5,1VCSTNN5DT,QJ5GSY0Y7O,8LI30OUMNO,ZWRZ4E4QN8,2R2001VJRH,XR4Q2992I2,Q7S1VG75BQ,IBSNP0S0KG,VCBMQ40FNN,QSHRIT977Q,5JGZSS2QPB,RL0U9FCRT9,JI8F4EY4CA,5XKR544N7Q,OI278Z7NYZ,D389NLF4BC,IVP4KUWQ7U,1ET6L0BAC9,0S4QQBJFDL,GXG5SM11ZX,7HN1DMLO5B,5YORL6N7NR,7PM5UTQP2R'.split(','):
        print('Pass!')
    else:
        print('Fail: Value Error')
    return
def ex2_1(generator_fib):
    target = [0]
    a, b = 0, 1
    while b < 10:
        target.append(b)
        a, b = b, a+b
    answer = list(generator_fib)
    if answer == target:
        print('Pass!')
    else:
        print('Fail: Value Error')
def ex2_2(student, rect1, rect2):
    if student.calculate(rect1, rect2) == (2*3 + 3*4):
        print('Pass')
    else:
        print('Value Error')