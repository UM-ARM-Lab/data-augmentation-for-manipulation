import itertools

spinners = {
    'dots': [
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏",
    ],
    'bars': '|\\-/',
}


class SynchronousSpinner:

    def __init__(self, message: str, spinner_name: str = 'dots'):
        self.message = message
        self.symbol_generator = itertools.cycle(spinners[spinner_name])

    def update(self):
        i = next(self.symbol_generator)
        print('\b' + i, end='', flush=True)

    def stop(self):
        print('\b', flush=True, end='')


def main():
    s = SynchronousSpinner("Loading")
    for i in range(100):
        from time import sleep

        sleep(0.1)
        s.update()
    s.stop()
    print("done!")


if __name__ == '__main__':
    main()
