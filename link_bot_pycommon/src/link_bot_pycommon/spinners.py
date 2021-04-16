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
        self.total_len = (len(self.message) + 2)
        self.backspaces = '\b' * self.total_len
        self.symbol_generator = itertools.cycle(spinners[spinner_name])

    def update(self):
        i = next(self.symbol_generator)
        print(f'{self.backspaces}{i} {self.message}', end='', flush=True)

    def stop(self):
        print('', flush=True)


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
