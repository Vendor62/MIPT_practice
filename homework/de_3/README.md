### Домашнее задание 2
Telegram бот на API Yandex GPT, собирает в логи следующую информацию:
- id пользователя
- время, когда пользователь произвёл любое действие с ботом
- тип действия (текст, команда)
- длина запроса пользователя
- длина ответа бота
- время обработки ботом
- текст запроса
- текст ответа

Бот сохраняет логи в csv, после того как наберётся 10 записей или после выполнения команды /backup конвертирует csv файл в xlsx и отправляет на Яндекс Диск, откуда файл забирает DataLens и строит дашборд со статистикой.

[Ссылка на дашборд](https://datalens.yandex/kqgvw8xcdnkk6)

Бот развёрнут на VM Yandex Cloud.

![](https://raw.githubusercontent.com/Vendor62/MIPT_practice/refs/heads/main/homework/de_3/media/01.png)

#### Структура репозитория
[Скрипт бота](https://github.com/Vendor62/MIPT_practice/blob/main/homework/de_3/bot_03.py)<br>
[Файл с логами](https://github.com/Vendor62/MIPT_practice/blob/main/homework/de_3/user_actions.csv)<br>
[Блокнот для чтения логов](https://github.com/Vendor62/MIPT_practice/blob/main/homework/de_3/read_logs.ipynb)<br>