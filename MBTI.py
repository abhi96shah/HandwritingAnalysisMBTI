import csv

row_count = 0

# loading csv file
with open('MBTI.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:

        row_count += 1
        # extr = 0
        # intr = 0
        # senr = 0
        # intur = 0
        # thinkr = 0
        # feelr = 0
        # judr = 0
        # perr = 0
        Mbti = {}
        # Ignoring first row
        if row_count == 1:
            print("ignored first row")
            continue

        else:
            Mbti = {'extr': 0, 'intr': 0, 'senr': 0, 'intur': 0,
                    'thinkr': 0, 'feelr': 0, 'judr': 0, 'perr': 0}
            print("computing for ", row[1])
            for i in range(2, 72):
                try:
                    if (i - 1) % 7 == 1:
                        if str(row[i][0]) == "a":
                            Mbti['extr'] += 1
                        elif str(row[i][0]) == "b":
                            Mbti['intr'] += 1

                    if (i - 1) % 7 == 2 or (i - 1) % 7 == 3:
                        if str(row[i][0]) == "a":
                            Mbti['senr'] += 1
                        elif str(row[i][0]) == "b":
                            Mbti['intur'] += 1

                    if (i - 1) % 7 == 4 or (i - 1) % 7 == 5:
                        if str(row[i][0]) == "a":
                            Mbti['thinkr'] += 1
                        elif str(row[i][0]) == "b":
                            Mbti['feelr'] += 1

                    if (i - 1) % 7 == 6 or (i - 1) % 7 == 0:
                        if str(row[i][0]) == "a":
                            Mbti['judr'] += 1
                        elif str(row[i][0]) == "b":
                            Mbti['perr'] += 1

                except:
                    continue

            if Mbti['extr'] != Mbti['intr']:
                if Mbti['extr'] > Mbti['intr']:
                    print("Extrovert")
                else:
                    print("Introvert")
            else:
                print("Can't Decide!")

            if Mbti['senr'] != Mbti['intur']:
                if Mbti['senr'] > Mbti['intur']:
                    print("Sensor")
                else:
                    print("Intuitive")
            else:
                print("Can't Decide!")

            if Mbti['thinkr'] != Mbti['feelr']:
                if Mbti['thinkr'] > Mbti['feelr']:
                    print("Thinker")
                else:
                    print("Feeler")
            else:
                print("Can't Decide!")

            if Mbti['judr'] != Mbti['perr']:
                if Mbti['judr'] > Mbti['perr']:
                    print("Judging")
                else:
                    print("Perceiving")
            else:
                print("Can't Decide!")
