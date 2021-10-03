# • временные метки, конкретные моменты времени
# • фиксированные периоды(январь 2007г. или весь 2010г)
# • временные интервалы, обозначаемые метками начала и конца(можно считать частными случаями интервалов)
import pandas as pd
import numpy as np


# Типы данных
from datetime import datetime
now = datetime.now()
print(now.year, now.month, now.day)


# Основы работы с временными рядами
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
         datetime(2011, 1, 7), datetime(2011, 1, 8),
         datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = pd.Series(np.random.randn(6), index=dates)


# Индексирование, выборка, подмножества
# передать только год или год и месяц
longer_ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
# Только 2001г
# longer_ts['2001']
# ///
# для объекта DataFrame, индексированного по строкам
dates = pd.date_range('1/1/2000', periods=100, freq='W–WED')
long_df = pd.DataFrame(np.random.randn(100, 4), index=dates, columns=['Colorado', 'Texas', 'New York', 'Ohio'])
# Только 5.2001
long_df.loc['5–2001']


# Диапазоны дат, частоты и сдвиг(постоянная чистота)
# преобразовать в ряд с частотой один день
resampler = ts.resample('D')
# ///
# Генерация диапазонов дат
index = pd.date_range('2012–04–01', '2012–06–01')  # По дефолту 1 день
# Генерация 20 дней
pd.date_range(start='2012–04–01', periods=20)
# границы для сгенерированного индекса по датам
# (BM - Последний рабочий день месяца)
pd.date_range('2000–01–01', '2000–12–01', freq='BM')
# ///
# Частоты и смещения дат
from pandas.tseries.offsets import Hour, Minute
# Частота в часах
pd.date_range('2000–01–01', '2000–01–03 23:59', freq='4h')
# в виде строки '1h30min'
pd.date_range('2000–01–01', periods=10, freq='1h30min')
# ///
# Даты, связанные с неделей месяца
# третья пятница каждого месяца
rng = pd.date_range('2012–01–01', '2012–09–01', freq='WOM–3FRI')
# ///
# Сдвиг данных (с опережением и с запаздыванием)
# Создать 4 значения со сдвигом на 1месяц
ts = pd.Series(np.random.randn(4), index=pd.date_range('1/1/2000', periods=4, freq='M'))


# Часовые пояса
import pytz
tz = pytz.timezone('America/New_York')
# Локализация и преобразование
zxc = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
tss = pd.Series(np.random.randn(len(zxc)), index=zxc)
# преобразования в другой часовой пояс
ts_utc = ts.tz_localize('UTC')
ts_utc.tz_convert('America/New_York')
# ///
# Операции над объектами Timestamp с учетом часового пояса
# локализовать, отдельные объекты Timestamp, включив информацию о часовом поясе и
# преобразовывать из одного пояса в другой
stamp = pd.Timestamp('2011–03–12 04:00')
stamp_utc = stamp.tz_localize('utc')
stamp_utc.tz_convert('America/New_York')
# задать часовой пояс при создании объекта Timestamp
stamp_moscow = pd.Timestamp('2011–03–12 04:00', tz='Europe/Moscow')
# ///
# Операции между датами из разных часовых поясов
rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2
print(result.index)
