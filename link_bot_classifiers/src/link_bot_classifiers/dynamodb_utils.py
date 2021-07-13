from colorama import Fore

primary_key = 'uuid'


def table(debug=False):
    if debug:
        input(Fore.RED + "Press enter to confirm you want to use the debugging database" + Fore.RESET)
        table_name = 'debugging'
    else:
        table_name = 'classifier-evaluation'
