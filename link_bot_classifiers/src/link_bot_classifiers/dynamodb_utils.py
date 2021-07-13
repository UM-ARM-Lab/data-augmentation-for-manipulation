from colorama import Fore

primary_key = 'uuid'

DEBUGGING = True

if DEBUGGING:
    input(Fore.RED + "Press enter to confirm you want to use the debugging database" + Fore.RESET)
    table_name = 'debugging'
else:
    table_name = 'classifier-evaluation'
