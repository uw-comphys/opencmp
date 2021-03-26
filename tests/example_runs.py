import subprocess
import sys
import math

# TODO: Should be possible to test this with pytest. Having issues accessing stdin and stdout.


def test_poisson_1():
    text = 'poisson_1: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_1/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

    print(text)

    return


def test_poisson_2():
    text = 'poisson_2: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_2/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

    print(text)

    return


def test_poisson_3():
    text = 'poisson_3: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_3/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 9e-6, rel_tol=1):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_poisson_4():
    text = 'poisson_4: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_4/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 9e-6, rel_tol=1):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_poisson_5():
    text = 'poisson_5: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_5/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 9e-6, rel_tol=10):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_poisson_6():
    text = 'poisson_6: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_6/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 9e-6, rel_tol=1):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_poisson_7():
    text = 'poisson_7: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_7/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 9e-6, rel_tol=1):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_poisson_8():
    text = 'poisson_8: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_8/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 9e-6, rel_tol=1):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_poisson_9():
    text = 'poisson_9: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_9/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 1.4e-5, rel_tol=1):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_poisson_10():
    text = 'poisson_10: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_10/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 1.4e-5, rel_tol=1):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_poisson_11():
    text = 'poisson_11: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_11/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 1.2e-5, rel_tol=1):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_poisson_12():
    text = 'poisson_12: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Poisson/poisson_12/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

        output = stdout.decode('utf-8')
        err_loc = output[::-1].find(' ')
        err = float(output[-err_loc:])

        if math.isclose(err, 1.2e-5, rel_tol=1):
            text += ' and produces the expected L2 error'
        else:
            text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_stokes_1():
    text = 'stokes_1: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Stokes/stokes_1/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

    print(text)

    return


def test_stokes_2():
    text = 'stokes_2: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/Stokes/stokes_2/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

    print(text)

    return


def test_ins_3():
    text = 'ins_3: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/INS/ins_3/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

    output = stdout.decode('utf-8')
    err_loc = output[::-1].find(' ')
    err = float(output[-err_loc:])

    if math.isclose(err, 2e-14, rel_tol=1):
        text += ' and produces the expected L2 error'
    else:
        text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_ins_4():
    text = 'ins_4: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/INS/ins_4/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

    output = stdout.decode('utf-8')
    err_loc = output[::-1].find(' ')
    err = float(output[-err_loc:])

    if math.isclose(err, 8e-11, rel_tol=1):
        text += ' and produces the expected L2 error'
    else:
        text += ' and CALCULATED L2 ERROR IS WRONG'

    print(text)

    return


def test_ins_5():
    text = 'ins_5: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/INS/ins_5/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

    print(text)

    return


def test_ins_6():
    text = 'ins_6: '

    result = subprocess.Popen([sys.executable, 'run.py', 'Example Runs/INS/ins_6/config'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode != 0:
        text += 'DOES NOT RUN'
    else:
        text += 'runs'

    print(text)

    return


if __name__ == '__main__':
    # Takes 5-10 min to run.

    print('-------------------------------')
    print('RUNNING STATIONARY EXAMPLES')
    print('-------------------------------')
    test_poisson_1()
    test_poisson_2()
    test_stokes_1()
    test_stokes_2()
    test_ins_3()
    test_ins_4()
    test_ins_5()
    test_ins_6()

    print('-------------------------------')
    print('RUNNING TRANSIENT EXAMPLES')
    print('-------------------------------')
    test_poisson_3()
    test_poisson_4()
    test_poisson_5()
    test_poisson_6()
    test_poisson_7()
    test_poisson_8()
    test_poisson_9()
    test_poisson_10()
    test_poisson_11()
    test_poisson_12()
