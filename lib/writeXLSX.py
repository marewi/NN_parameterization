import xlsxwriter

def writeXLSX(self):
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook('min_losses.xlsx')
    worksheet = workbook.add_worksheet()

    # # Some data we want to write to the worksheet.
    # expenses = (
    #     ['Rent', 1000],
    #     ['Gas',   100],
    #     ['Food',  300],
    #     ['Gym',    50],
    # )

    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0

    # Iterate over the data and write it out row by row.
    for set, loss in (self):
        worksheet.write(row, col    , row)
        worksheet.write(row, col + 1, str(set))
        worksheet.write(row, col + 2, str(loss))
        row += 1

    # Write a total using a formula.
    # worksheet.write(row, 0, 'Total')
    # worksheet.write(row, 1, '=SUM(B1:B4)')

    workbook.close()