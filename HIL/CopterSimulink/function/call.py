import filterData

t = filterData.RecordPosAndDistance()
t.updateMeanStd()
t.insertNewXYR(1,2,3)
t.printDataX()