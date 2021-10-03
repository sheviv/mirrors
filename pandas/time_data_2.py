# • временные метки, конкретные моменты времени
# • фиксированные периоды(январь 2007г. или весь 2010г)
# • временные интервалы, обозначаемые метками начала и конца(можно считать частными случаями интервалов)
import pandas as pd
import numpy as np
import matplotlib as plt


# Периоды и арифметика периодов
# Периоды – это промежутки времени: дни, месяцы, кварталы, годы(класс Period)
p = pd.Period(2007, freq='A–DEC')  # промежуток времени от 1 января 2007 года до 31 декабря 2007 года включительно.
# Сложение и вычитание периода и целого числа(сдвиг на величину кратную частоте периода)
p + 5  # Period('2012', 'A–DEC')
p - 2  # Period('2005', 'A–DEC')
# ///
# Регулярные диапазоны периодов
rng = pd.period_range('2000–01–01', '2000–06–30', freq='M')  # PeriodIndex(['2000–01', '2000–02', '2000–03', '2000–04', '2000–05', '2000–06'], dtype='period[M]', freq='M')
# ///
# Преобразование частоты периода
# годовой период преобразовать в месячный, начинающийся или заканчивающийся на границе года
p = pd.Period('2007', freq='A–DEC')
p.asfreq('M', how='start')
p.asfreq('M', how='end')
# ///
# Для финансового года, заканчивающегося в любом месяце, кроме декабря, месячные подпериоды
p = pd.Period('2007', freq='A–JUN')
p.asfreq('M', 'start')
p.asfreq('M', 'end')

# Квартальная частота периода
# 12 возможных значений квартальной частоты – от Q–JAN до Q–DEC
p = pd.Period('2012Q4', freq='Q–JAN')
p.asfreq('D', 'start')
p.asfreq('D', 'end')
# ///
# временная метка для момента «4 часа пополудни предпоследнего рабочего дня квартала»
p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
p4pm.to_timestamp()
# ///
# генерация квартальных диапазонов
rng = pd.period_range('2011Q3', '2012Q4', freq='Q–JAN')
ts = pd.Series(np.arange(len(rng)), index=rng)
new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
ts.index = new_rng.to_timestamp()

# Преобразование временных меток в периоды и обратно
# Объекты Series и DataFrame, индексированные временными метками преобразовать в периоды
rng = pd.date_range('2000–01–01', periods=3, freq='M')
ts = pd.Series(np.random.randn(3), index=rng)
pts = ts.to_period()


# Передискретизация и преобразование частоты
# процесс изменения частоты временного ряда
rng = pd.date_range('2000–01–01', periods=100, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts.resample('M').mean()  # ts.resample('M', kind='period').mean()
# ///
# Понижающая передискретизация
# • какой конец интервала будет включаться;
# • помечать ли агрегированный интервал меткой его начала или конца.
# данные с частотой одна минута:
rng = pd.date_range('1/1/2000', periods=12, freq='T')
ts = pd.Series(np.arange(12), index=rng)
# агрегировать данные в пятиминутные группы, или столбики, вычислив сумму по каждой группе
ts.resample('5min', how='sum')
ts.resample('5min', closed='right').sum()  # closed='right' - включается правый конец интервала
# Результирующий временной ряд помечен временными метками, соответствующими левым концам интервалов
ts.resample('5min', closed='right', label='right').sum()
# вычесть одну секунду из правого конца, чтобы было понятнее, к какому интервалу относится временная метка
ts.resample('5min', closed='right', label='right', loffset='–1s').sum()
# ///
# Передискретизация OHLC
# четыре значения для каждого интервала:
# первое (открытие – open)
# последнее (закрытие – close)
# максимальное (high)
# минимальное (low)
ts.resample('5min').ohlc()

# Повышающая передискретизация и интерполяция
frame = pd.DataFrame(np.random.randn(2, 4),
                     index=pd.date_range('1/1/2000', periods=2,
                                         freq='W–WED'),
                     columns=['Colorado', 'Texas', 'New York', 'Ohio'])
# перейти к более высокой частоте без агрегирования
df_daily = frame.resample('D').asfreq()
# ///
# Передискретизация периодов
annual_frame = frame.resample('A–DEC').mean()
annual_frame.resample('Q–DEC').ffill()
annual_frame.resample('Q–DEC', convention='end').ffill()


# Скользящие оконные функции
# для операций с временными рядами, – статистические и иные функции
# загрузить временной ряд и передискретизировать на частоту «рабочий день»
close_px_all = pd.read_csv('examples/stock_px_2.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B').ffill()
# rolling(250) - создает объект(допускает группировку по скользящему окну шириной 250 дней), - средние котировки акций Apple
# в скользящем окне шириной 250 дней.
appl_std250 = close_px.AAPL.rolling(250, min_periods=10).std()
appl_std250.plot()
# Среднее с расширяющимся окном для временного ряда apple_std250
expanding_mean = appl_std250.expanding().mean()
# преобразование применяется к каждому столбцу
close_px.rolling(60).mean().plot(logy=True)
# скользящее среднее за 20 дней
close_px.rolling('20D').mean()

# Экспоненциально взвешенные функции
# постоянный коэффициент затухания, чтобы повысить вес последних наблюдений
# скользящее среднее котировок акций Apple за 60 дней сравнивается с экспоненциально взвешенным скользящим средним для span=60
aapl_px = close_px.AAPL['2006':'2007']
ma60 = aapl_px.rolling(30, min_periods=20).mean()
ewma60 = aapl_px.ewm(span=30).mean()
ma60.plot(style='k--', label='Simple MA')
ewma60.plot(style='k-', label='EW MA')
plt.legend()

# Бинарные скользящие оконные функции
# корреляции и ковариации, необходимы два временных ряда
# вычислить относительные изменения в процентах для всего нашего временного ряда
spx_px = close_px_all['SPX']
spx_rets = spx_px.pct_change()
returns = close_px.pct_change()
corr = returns.AAPL.rolling(125, min_periods=100).corr(spx_rets)
corr.plot()
# вычислить корреляцию индекса S&P 500
corr = returns.rolling(125, min_periods=100).corr(spx_rets)
corr.plot()

# Скользящие оконные функции, определенные пользователем
from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
result = returns.AAPL.rolling(250).apply(score_at_2percent)
result.plot()

