primary_key = 'uuid'

DEBUGGING = True

if DEBUGGING:
    input("Press enter to confirm you want to use the debugging database")
    table_name = 'debugging'
else:
    table_name = 'classifier-evaluation'
