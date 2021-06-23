import shutil


def get_width():
    width, _ = shutil.get_terminal_size((80, 20))  # pass fallback
    return width


def banner(c: str, msg: str):
    assert len(c) == 1
    width = get_width()
    s = int((width - len(msg) - 2) / 2)
    return f'{c * s} {msg} {c * s}'


def stars(msg):
    return banner('*', msg)


def equals(msg):
    return banner('=', msg)


def blocks(msg):
    return banner('━', msg)
