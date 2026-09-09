"""Quick portable correctness test: mpirun -n 2 python Jean-Zay/check.py."""
from worker import entry

if __name__ == '__main__':
    entry(check_only=True)
