# Change CSV to JSON
import csv
import json
contents = open('name.csv').readlines()
json.dumps(list(csv.reader(contents)))
#
