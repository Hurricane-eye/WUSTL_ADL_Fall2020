import csv


def make_record_in_csv(result):

    with open("test.csv", "r") as test_file:
        reader =csv.DictReader(test_file)

        submit_file = open("submit.csv", "w")
        fieldnames = ["id", "stable"]
        writer = csv.DictWriter(submit_file, fieldnames)
        writer.writeheader()

        reader.fieldnames
        i = 1
        for row in reader:
            writer.writerow({"id": row["id"], "stable": str(int(result[i]))})
            i += 1

