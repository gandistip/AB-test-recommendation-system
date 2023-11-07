#!/usr/bin/env python
# coding: utf-8

# # Анализ результатов A/B-теста рекомендательной системы

# ## Вводная информация

# ### Цель 

# - Провести оценку результатов A/B-теста изменений, связанных с внедрением улучшенной рекомендательной системы;

# ### Задачи

# - Оценить корректность проведения теста
#   - проверить пересечение тестовой аудитории с конкурирующим тестом
#   - проверить совпадение теста и маркетинговых событий
#   - проверить другие проблемы временных границ теста
# - Проанализировать результаты теста

# ### Техническое задание

# - Название теста: `recommender_system_test`
# - Группы: А — контрольная, B — новая платёжная воронка
# - Дата запуска: 2020-12-07
# - Дата остановки набора новых пользователей: 2020-12-21
# - Дата остановки: 2021-01-04
# - Аудитория: 15% новых пользователей из региона EU
# - Назначение теста: тестирование изменений, связанных с внедрением улучшенной рекомендательной системы
# - Ожидаемое количество участников теста: 6000
# - Ожидаемый эффект: за 14 дней с момента регистрации пользователи покажут улучшение каждой метрики не менее, чем на 10%
#     - `product_page` - конверсии в просмотр карточек товаров
#     - `product_cart` - просмотры корзины
#     - `purchase` - покупки.

# ### Описание данных

# **marketing_events** — календарь маркетинговых событий на 2020 год:
# - `name` — название маркетингового события
# - `regions` — регионы, в которых будет проводиться рекламная кампания
# - `start_dt` — дата начала кампании
# - `finish_dt` — дата завершения кампании
# 
# **users** — пользователи, зарегистрировавшиеся 07.12.2020 - 21.12.2020:
# - `user_id` — идентификатор пользователя
# - `first_date` — дата регистрации
# - `region` — регион пользователя
# - `device` — устройство, с которого происходила регистрация
# 
# **events** — действия новых пользователей 07.12.2020 - 04.01.2021:
# - `user_id` — идентификатор пользователя
# - `event_dt` — дата и время покупки
# - `event_name` — тип события
# - `details` — стоимость покупки в долларах (только для `purchase`)
# 
# **participants** — таблица участников тестов:
# - `user_id` — идентификатор пользователя
# - `ab_test` — название теста
# - `group` — группа пользователя

# ### Использованные библиотеки

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from plotly import graph_objects as go
import math as mth
import datetime


# ## Обзор и подготовка данных

# ### Загрузка и отображенние данных

# In[2]:


marketing_events, users, events, participants = (
    pd.read_csv('/datasets/ab_project_marketing_events.csv'),
    pd.read_csv('/datasets/final_ab_new_users.csv'),
    pd.read_csv('/datasets/final_ab_events.csv'),
    pd.read_csv('/datasets/final_ab_participants.csv')
)
all_df = [
    ['\033[1m Маркетинговые события \033[0m- marketing_events', marketing_events,],
    ['\033[1m Пользователи \033[0m- users',                     users],
    ['\033[1m Действия \033[0m- events',                        events],
    ['\033[1m Участники тестов \033[0m- participants',          participants]
]


# In[3]:


for i,j in all_df:
  print()
  print(i)
  display(j.head(3))


# ### Преобразование типов и пропуски в данных

# In[4]:


for i,j in all_df:
  print()
  print(i)
  display(j.info())


# Пропусков значений в данных нет, кроме доп.признака `details` таблицы `events`, который присутствует только для событий `purchase`. 

# Изменим тип данных полей с датами:

# In[5]:


users['first_date'] = pd.to_datetime(users['first_date'], format="%Y-%m-%d")
events['event_dt'] = pd.to_datetime(events['event_dt'], format="%Y-%m-%d")
marketing_events['start_dt'] = pd.to_datetime(marketing_events['start_dt'], format="%Y-%m-%d")
marketing_events['finish_dt'] = pd.to_datetime(marketing_events['finish_dt'], format="%Y-%m-%d")


# ### Проверка наличия дубликатов

# In[6]:


for i,j in all_df:
  print()
  print(i)
  print('Полных дубликатов:', j[j.duplicated()].size)


# Полных дубликатов в датасетах нет.

# In[7]:


for i,j in all_df[1:]:
  print()
  print(i)
  print('Нет повторяющихся ID пользователей:', j.user_id.nunique() == j.user_id.size)


# Повторяющихся пользователей в `user` нет и это хорошо. В остальных таблицах повтоящиеся пользователи есть и это логично, пользователи могут совершать несколько действий и могут пересекаться в разных тестах.

# ## Оценка корректность проведения теста

# ### Проверка соответствия данных требованиям технического задания

# **Пункт ТЗ:** Название теста: `recommender_system_test`. (Соблюден)

# Удалим из таблицы `participants` все не относящееся к `recommender_system_test`:

# In[8]:


partic = participants.query('ab_test == "recommender_system_test"').drop('ab_test', axis=1)


# **Пункт ТЗ:** Дата остановки набора новых пользователей: 2020-12-21. (Соблюден)

# Удалим значения превыщающие эту дату и проверим результат:

# In[9]:


users = users.query('first_date < "2020-12-22"')
print('Даты регистрации пользователей:')
print(users.first_date.min(), '- начальная')
print(users.first_date.max(), '- конечная')


# **Пункт ТЗ:** Дата запуска: 2020-12-07, дата остановки: 2021-01-04. **(Не соблюден)**

# In[10]:


print('Период имеющихся событий:')
print(events.event_dt.min(), '- первое')
print(events.event_dt.max(), '- последнее')


# Тест закончился на 5 дней раньше, чем обозначено в ТЗ, соответственно пользователи зарегистрированные после 16 декабря не наберут требуемых в ТЗ 14 дней использования ресурса.

# **Пункт ТЗ:** Аудитория: 15% новых пользователей из региона EU. (Соблюден)

# Оставим только пользователей из региона EU:

# In[11]:


users = users.query('region == "EU"').drop('region', axis=1)


# Объеденим таблицы `users`, `partic`, `events`:

# In[12]:


df = users.merge(partic, on='user_id', how='inner').merge(events, on='user_id', how='left').drop('device',axis=1)
df = df.rename(columns={'user_id':'user', 'event_name':'event', 'details':'revenue'})
df.head(5)


# In[13]:


print('Новых пользователей из региона EU:')
df.user.nunique() / users.user_id.nunique()


# **Пункт ТЗ:** Ожидаемое количество участников теста: 6000. (Соблюден)

# In[14]:


print('Пользователей:', df.user.nunique())


# **Пункт ТЗ (не соблюден):** Ожидаемый эффект: за 14 дней с момента регистрации пользователи покажут улучшение конверсии не менее, чем на 10% для: 
# - просмотров карточек товаров,
# - просмотров корзины,
# - покупок.

# Оставим только те события которые совершены в первые 14 дней с момента регистрации пользователей, тем самым за одно исключим пользователей не имеющих ни одного события:

# In[15]:


df['date_delta'] = df['event_dt'] - df['first_date']
df = df[df['date_delta'] <= pd.Timedelta(days=14)].drop('date_delta', axis=1).copy()
print('Пользователей совершавших события на протяжении 14 дней после регистрации:', df.user.nunique())


# Количество пользоватей после удовлетворения данного пункта ТЗ существенно уменьшилось.

# Посчитаем конверсии на требуемых этапах:

# In[16]:


funnel = df.pivot_table(index='event', columns='group', values='user', aggfunc=['nunique'])
funnel.columns = funnel.columns.droplevel(0)
funnel = funnel.reset_index()
funnel['funnel'] = [1,3,2,4]
funnel = funnel.set_index('funnel').reset_index().sort_values('funnel')
funnel['conv_A'] = (funnel['A'].shift(periods=-1) / funnel['A']).shift(periods=1).round(2)
funnel['conv_B'] = (funnel['B'].shift(periods=-1) / funnel['B']).shift(periods=1).round(2)
print('Количество и доля пользователей воронки событий по группам:')
funnel


# Как видно пользователи не показали улучшение конверсии на 10% для: просмотров карточек товаров, просмотров корзины, покупок.

# ### Проверка пересечений с маркетинговыми и другими активностями

# In[17]:


marketing_events.query('"2020-12-07" <= start_dt <= "2020-12-30"')


# Есть пересечение нашего теста по времени и региону проведения с рекламной компанией:
# - `Christmas&New Year Promo` пересекается последние 6 дней нашего теста.
# 
# По имеющимся данным мы не можем сказать учавствовали ли пользователи нашего теста в этой рекламной компании.
# 
# Также тест попадает в общую предпраздничную покупательскую активность, что тоже может негативно отразиться на качестве АВ-теста.

# ### Проверка пересечений с конкурирующим тестом

# In[18]:


interface_eu_test_users = participants.query('ab_test == "interface_eu_test"').user_id.unique()
print('Пересечений пользователей нашего теста с конкурирующим тестом:')
df.query('user in @interface_eu_test_users').user.nunique()


# Пересечения есть, и возможно конкурирующий тест оказывает влияние на наш тест и т.к. мы не знаем о деталях конкурирующего теста, то уберем из данных учавствующих в обоих тестах:

# In[19]:


df = df.query('user not in @interface_eu_test_users')


# ### Проверка пересечений участвующих в двух группах теста одновременно

# In[20]:


a_users = df.query('group == "A"').user.unique()
print('Пересечений пользователей в групах А и В:')
df.query('group == "B" and user in @a_users').user.nunique()


# ### Проверка равномерности распределения по тестовым группам

# In[21]:


t = df.groupby('group').agg({'count','nunique'})['user'].rename(columns={'nunique':'user','count':'event'})
t['user_%'] = (t['user'] / t['user'].sum() * 100).round(1)
t['event_%'] = (t['event'] / t['event'].sum() * 100).round(1)
print('Соотношение количества и долей событий и пользователей:')
t


# Пользователей в группах 1/3 (один к трем) в пользу группы А. Распределение пользователей по группа крайне **не равномерное** с кратным превышением доли в группе А. Также можно заметить, что соотношение активности пользователей тоже выше в пользу группы А.

# ## Исследовательский анализ данных

# ### Распределения количества событий на пользователя в выборках

# Посчитаем среднее количество событий на одного пользователя по каждой группе:

# In[22]:


df_a = df.query('group == "A"')
df_b = df.query('group == "B"')
user_event_a = df_a.groupby('user').count()['event']
user_event_b = df_b.groupby('user').count()['event']
print('Количество событий на одного пользователя:')
print(user_event_a.mean().round(1), '- в группе А')
print(user_event_b.mean().round(1), '- в группе B')
print('на', (100 -(user_event_b.mean() / user_event_a.mean() *100)).round(1),'% пользователи активнее в группе А')


# In[23]:


plt.figure(figsize=(10, 5))
user_event_a.hist(bins=user_event_a.max())
user_event_b.hist(bins=user_event_b.max())
plt.legend(labels = df.group.unique())
plt.ylabel('Пользователей')
plt.xlabel('Событий')
plt.show()


# График подтверждает, что пользователи из группы В совершают меньше действий чем в А.
# 
# 6 действий совершило больше всего пользователей, но в целом распределение аномальное и имеет явные пробелы: 5 и 7 действий совершает аномально малое количество пользователей не согласующееся с нормальным распределением.

# ### Распределение числа событий в выборках по дням
# 

# In[24]:


plt.figure(figsize=(15, 5))
sns.countplot(data=df, hue='group', x=df['event_dt'].sort_values().dt.date)
plt.xticks(rotation = 45)
plt.show()


# С 14 числа заметен резкий рост активности пользователей в группе А достигающий пика 21 числа, вероятно это связано с трендовой предновогодней активностью пользователей.
# 
# Группа В не показывает значительного роста активности и уступает по активности во все дни кроме 7 числа - первого дня теста.

# ### Изменение конверсии в воронке в выборках на разных этапах
# 

# In[25]:


print('Пользователей всего:', df.user.nunique())
print('Пользователей имеющих событие login:', df.query('event == "login"').user.nunique())


# Все кроме одного пользователя имеют событие `login`, что логично и говорит о том, что это первая ступень воронки и без нее невозможно сделать `покупку`, а покупка `purchase` в свою очередь это логичная последняя ступень воронки. Рассчитаем конверсию на каждом этапе воронки:

# In[26]:


funel = df.pivot_table(index='event', columns='group', values='user', aggfunc=['nunique'])
funel.columns = funel.columns.droplevel(0)
funel = funel.reset_index()
funel['funel'] = [1,3,2,4]
funel = funel.set_index('funel').reset_index().sort_values('funel')
funel['conv_A'] = (funel['A'].shift(periods=-1) / funel['A']).shift(periods=1).round(2)
funel['conv_B'] = (funel['B'].shift(periods=-1) / funel['B']).shift(periods=1).round(2)
funel['conv_A_total'] = (funel['A'] / df_a['user'].nunique()).round(2)
funel['conv_B_total'] = (funel['B'] / df_b['user'].nunique()).round(2)
print('Количество и доля пользователей воронки событий по группам:')
funel


# Отличие конверсии в группе В от группы А:
# - регистрация => карточка товара - хуже на 9%
# - регистрация => корзина - хуже на 2%
# - регистрация => покупка - хуже на 3%
# - карточка товара => корзина - лучше на 3%
# - корзина => покупка - без отличий

# In[27]:


fig = go.Figure()
fig.add_trace(go.Funnel(
    name = 'Группа A',
    y = funel[['event','A']].event,
    x = funel[['event','A']].A,
    textinfo = "value + percent initial + percent previous"))
fig.add_trace(go.Funnel(
    name = 'Группа B',
    y = funel[['event','B']].event,
    x = funel[['event','B']].B,
    textinfo = "value + percent initial + percent previous"))
fig.show()


# ### Учет особенностей данных перед A/B-тестированием

# Проверим достаточно ли длился тест и стабилизировались ли к концу теста кумулятивные метрики конверсии по обоим группам, подготовим данные и визуализируем графики:

# In[28]:


# создаем таблицу с заказами
orders = df.dropna(subset=['revenue']).drop(['first_date','event'],axis=1).reset_index()           .rename(columns={'user':'visitorId','event_dt':'date','index':'transactionId'})
orders['date'] = orders['date'].dt.date

# создаем таблицу с посетителями по дням и группам
visitors = df[['event_dt','group']].rename(columns={'event_dt':'date'})
visitors['date'] = visitors['date'].dt.date
visitors = visitors.groupby(['date','group']).agg({'date':'count'}).rename(columns={'date':'visitors'}).reset_index()

# создаем массив уникальных пар значений дат и групп теста
datesGroups = orders[['date','group']].drop_duplicates()

# получаем агрегированные кумулятивные по дням данные о заказах 
ordersAggregated = datesGroups.apply(
  lambda x: orders[np.logical_and(orders['date'] <= x['date'], orders['group'] == x['group'])]\
  .agg({'date': 'max', 'group': 'max', 'transactionId': 'nunique', 'visitorId': 'nunique', 'revenue': 'sum'}), axis=1)\
  .sort_values(by=['date','group'])

# получаем агрегированные кумулятивные по дням данные о посетителях 
visitorsAggregated = datesGroups.apply(
  lambda x: visitors[np.logical_and(visitors['date'] <= x['date'], visitors['group'] == x['group'])]\
  .agg({'date': 'max', 'group': 'max', 'visitors': 'sum'}), axis=1)\
  .sort_values(by=['date','group'])

# объединяем кумулятивные данные в одной таблице
cumulativeData = ordersAggregated.merge(visitorsAggregated, left_on=['date', 'group'], right_on=['date', 'group'])
cumulativeData.columns = ['date', 'group', 'orders', 'buyers', 'revenue', 'visitors']

# датафрейм с кумулятивным количеством заказов и кумулятивной выручкой по дням в группе А и В
cumulativeRevenueA = cumulativeData[cumulativeData['group']=='A'][['date','revenue', 'orders']]
cumulativeRevenueB = cumulativeData[cumulativeData['group']=='B'][['date','revenue', 'orders']]


# In[29]:


# считаем кумулятивную конверсию
cumulativeData['conversion'] = cumulativeData['orders'] / cumulativeData['visitors']
cumulativeDataA = cumulativeData[cumulativeData['group']=='A']
cumulativeDataB = cumulativeData[cumulativeData['group']=='B']

plt.figure(figsize = (10, 5))
plt.plot(cumulativeDataA['date'], cumulativeDataA['conversion'], label='A')
plt.plot(cumulativeDataB['date'], cumulativeDataB['conversion'], label='B')
plt.title('Кумулятивная конверсия из регистрации в покупки')
plt.ylabel('Конверсия')
plt.legend()
plt.xticks(rotation = 45)
plt.show()


# Последнюю неделю теста кумулятивные **метрики конверсии стабильны**, значит длительность теста достаточна.

# Проверим имеются ли в данных значительные выбросы значений, построим распределение по пользователям `количества событий`, `заказов` и `стоимости заказов`:

# In[30]:


user_event = df.groupby('user').count()['event']
plt.title('Распределение количества событий по пользователям')
plt.scatter(pd.Series(range(0, user_event.size)), user_event.values)
plt.ylabel('Событий')
plt.xlabel('Пользователей')
plt.show()

user_purchase = df.query('event == "purchase"').groupby('user').count()['event']
plt.title('Распределение количества заказов по пользователям')
plt.scatter(pd.Series(range(0, user_purchase.size)), user_purchase.values)
plt.ylabel('Заказов')
plt.xlabel('Пользователей')
plt.show()

plt.scatter(pd.Series(range(0, len(orders['revenue']))), orders['revenue'])
plt.title('Распределение стоимости заказа по пользователям')
plt.ylabel('Стоимость заказа')
plt.xlabel('Пользователей')
plt.show()


# Сильно выбивающихся из общего распределения значений нет.

# ## Оцените результаты A/B-тестирования

# ### Результаты A/В-тестирования

# Построим графики кумулятивной выручки и среднего чека группы А и В:

# In[31]:


plt.figure(figsize = (10, 5))
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue'], label='A')
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue'], label='B')
plt.title('Кумулятивная выручка')
plt.ylabel('Выручка')
plt.legend()
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize = (10, 5))
plt.plot(cumulativeRevenueA['date'], cumulativeRevenueA['revenue']/cumulativeRevenueA['orders'], label='A')
plt.plot(cumulativeRevenueB['date'], cumulativeRevenueB['revenue']/cumulativeRevenueB['orders'], label='B')
plt.title('Кумулятивный средний чек')
plt.ylabel('Средний чек')
plt.legend()
plt.xticks(rotation = 45)
plt.show()


# Средний чек у группы А стабилен на протяжении всего теста. А средний чек у группы В значительно вырос в период с 12 по 15 число и стабилизировался.
# 
# Метрики кумулятивной выручки и среднего чека в группе В ни разу **не превысили** показатели группы А за все время теста.

# Построим график изменения кумулятивной конверсии из регистрации в покупки группы B к группе A:

# In[32]:


mergedCumulativeConversions = cumulativeDataA[['date','conversion']].merge(cumulativeDataB[['date','conversion']],
  left_on='date', right_on='date', how='left', suffixes=['A', 'B'])

plt.figure(figsize = (10, 5))
plt.plot(mergedCumulativeConversions['date'],
  mergedCumulativeConversions['conversionB'] / mergedCumulativeConversions['conversionA'] - 1,
  label="Относительный прирост конверсии группы B относительно группы A")
plt.title('Изменения кумулятивной конверсии из регистрации в покупки группы B к группе A')
plt.ylabel('Доля изменения')
plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=0.1, color='grey', linestyle='--')
plt.xticks(rotation = 45)
plt.show()


# Пользователи В группы показали **ухудшение** кумулятивной конверсии в покупки более чем на 5%.

# ### Проверьте статистическую разницу долей z-критерием.

# Будем сравнивать доли пользователей совершивших различные действия между контрольной группой А и тестовой группой В:
# - **Гипотеза нулевая Н0**: между долями нет значимой разницы 
# - **Гипотеза альтернативная Н1**: между долями возможно есть статистически значимая разница

# Напишем функцию проверки статистической разницы долей Z-критерием. Проверка множественная с несколькими событиями, значит необходимо изспользовать поправку к требуемому уровню статистической значимости alpha, используем самую популярную - Бонферрони:

# In[33]:


def equal_share(group1, group2, event_name, alpha, bonferroni):
  event_group_user = df.query('event == @event_name').groupby('group').agg({'user':'nunique'})
  group_user = df.groupby('group').agg({'user':'nunique'})
  print()
  print('Событие:', event_name)

  # критический уровень статистической значимости c попровкой Бонферрони
  bonferroni_alpha = alpha / bonferroni
  # пропорция успехов в группах и комбинированном датасете:
  p1 = event_group_user.loc[group1, 'user'].mean() / group_user.loc[group1, 'user'].mean()
  p2 = event_group_user.loc[group2, 'user'] / group_user.loc[group2, 'user']
  p_combined = (event_group_user.loc[group1, 'user'].mean() + event_group_user.loc[group2, 'user'])    / (group_user.loc[group1, 'user'].mean() + group_user.loc[group2, 'user'])
  # разница пропорций в датасетах
  difference = p1 - p2
  # считаем статистику в ст.отклонениях стандартного нормального распределения
  z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1/group_user.loc[group1, 'user'].mean() + 1/group_user.loc[group2, 'user']))
  # задаем стандартное нормальное распределение (среднее 0, ст.отклонение 1) и посчитаем p-value
  distr = st.norm(0, 1) 
  p_value = (1 - distr.cdf(abs(z_value))) * 2
  print('p-значение: ', p_value)
  # сравним p-value с уровнем стат.значимости
  if p_value < bonferroni_alpha: 
      print('Отвергаем нулевую гипотезу: между долями есть значимая разница')
  else:
      print('Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными')


# Запустим проверку равенства долей пользователей z-критерием с помощью нашей функции по каждому событию, alpha установим 0.05:

# In[34]:


events_test = df.query('event != "login"')['event'].unique()
for i in events_test:
  equal_share('A', 'B', i, 0.05, len(events_test))


# Разница в пропорциях при указанных размерах выборок достаточна, чтобы говорить о статистически значимом различии только для конверсии из `регистрации` в `просмотр карточки товара`.

# ## Выводы

# По этапу исследовательского анализа данных:
# - Количество событий на одного пользователя: 6.9 - в группе А, 5.4 - в группе B, на 21.6 % пользователи активнее в группе А.
# - С 14 числа заметен резкий рост активности пользователей в группе А достигающий пика 21 декабря.
# - Группа В не показывает значительного роста активности пользователей и уступает по активности во все дни кроме 7 декабря - первого дня теста.
# - Отличие конверсии в группе В от группы А:
#  - регистрация => карточка товара - хуже на 9%
#  - регистрация => корзина - хуже на 2%
#  - регистрация => покупка - хуже на 3%
#  - карточка товара => корзина - лучше на 3%
#  - корзина => покупка - без отличий
# - Метрики кумулятивной выручки и среднего чека в группе В ни разу не превысили показателей группы А за все время теста.
# - Средний чек у группы А стабилен на протяжении всего теста. А средний чек у группы В значительно вырос в период с 12 по 15 декабря и стабилизировался.
# - Исследовательский анализ показал, что новая рекомендательная система негативно влияет на показатели конверсии и выручки.
# 
# По проведённой оценке результатов A/B-тестирования:
# - Разница в пропорциях при указанных размерах выборок достаточна, чтобы говорить о статистически значимом различии для конверсии из регистрации в карточки товара, которая в группе В ниже чем в А на 9%.
# - Отсутствует статистически значимое различие конверсии в покупки и перехода в корзину между группами.
# 
# Общее заключение о корректности проведения теста:
# - В целом на основании теста нет возможности сделать статистически обоснованных и ценных для бизнеса выводов.
# - Неравномерное распределение пользователей между контрольно и тестовой группой различается почти на 300%.
# - Пересечение теста с периодом сильного изменения поведения пользователей в период предновогодней активности.
# - Раннее завершение теста и связанное с этим недобором пользователей с 2х-недельной активностью после регистрации.
# - Пересечение теста с конкурирующим тестом, что тоже снижает количество пользователей для анализа.
# - Итогое количество оставшихся пользователей для анализа теста 2594, что существенно меньше заявленных 6000 в ТЗ.
# 
