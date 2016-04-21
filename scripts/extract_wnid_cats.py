#!/usr/bin/env python3

import scipy.io

def main():
  meta_mat = scipy.io.loadmat("meta.mat")
  print(meta_mat["synsets"].shape)
  #print(meta_mat["synsets"])

  with open("raw_cats_db.csv", "w") as f:
    print("wnid,raw_cat", file=f)
    for row in meta_mat["synsets"]:
      #print(row)
      #print(row[0])
      raw_cat = row[0][0][0][0]
      wnid = row[0][1][0]
      if raw_cat >= 1 and raw_cat <= 1000:
        print("{},{}".format(wnid, raw_cat), file=f)

if __name__ == "__main__":
  main()
