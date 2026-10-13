from files import stem

assert stem("report.txt") == "report", stem("report.txt")
assert stem("archive.tar.gz") == "archive.tar", stem("archive.tar.gz")
print("ok")
