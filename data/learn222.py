import pandas as pd


def xlsx_to_csv_pd():
    data_xls = pd.read_excel('step_2021_300_14_result.xls', index_col=0)
    data_xls.to_csv('1.csv', encoding='utf-8')

if __name__ == '__main__':
    xlsx_to_csv_pd()
