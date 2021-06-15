import multiprocessing
from multiprocessing import Pool, Process


def _foo(x):
    return x + 2


def prefetch():
    pool = Pool(processes=2)
    mylist = list(range(6))
    print("calling imap")
    print(list(pool.imap_unordered(_foo, mylist)))


# prefetch_process = Process(target=prefetch, args=(pool,))
prefetch_process = Process(target=prefetch)
prefetch_process.start()

# print("calling imap")
# mylist = list(range(6))
# print(list(pool.imap_unordered(_foo, mylist)))
