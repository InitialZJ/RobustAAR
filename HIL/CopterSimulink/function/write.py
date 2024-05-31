import xlwt
f = xlwt.Workbook('encoding = utf-8') #设置工作簿编码
sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) #创建sheet工作表
list1 = [1,3,4,6,8,10]#要写入的列表的值
for i in range(len(list1)):
    sheet1.write(i,0,list1[i]) #写入数据参数对应 行, 列, 值
f.save('text.xls')#保存.xls到当前工作目录
