# Диагностический разбор примеров

Рассмотрены 30 контекстов, по 10 из каждого дампа, и по одному выбранному факту
на контекст. Выборка воспроизводится с seed 42: экстремальные длины, редкие отношения,
структурные сигналы и случайное дополнение. ID и причины отбора находятся в
[diagnostic_sample.json](tables/diagnostic_sample.json), выбранные факты с qualifiers —
в [diagnostic_facts.json](tables/diagnostic_facts.json).

Ниже — качественный разбор по исходному тексту, не независимая человеческая
разметка и не оценка precision/recall корпуса. Не проверялись все факты каждого
контекста, полнота извлечения и внешняя истинность утверждений. Выборка намеренно
включает подозрительные случаи, поэтому доли категорий нельзя обобщать на датасет.

H — HotpotQA validation, M — MuSiQue-Ans dev, G — MuSiQue-Full train.
Полные тексты доступны локально в `data/processed/eda/diagnostic_contexts.jsonl`;
источники и лицензии приведены в [отчёте сопоставления](../source_matching.md).

| № | Контекст | Выбранный факт | Наблюдение по тексту и действие |
|---|---|---|---|
| 1 | `H/5ab7f97a5542991d322237ef/3` | Nashville Terminal Subdivision → has part(s) → 5 sections | Есть «broken up into 5 sections all in Nashville». Количество подтверждается, но объект описывает число частей, а не конкретный узел-часть. Согласовать контракт количества. |
| 2 | `H/5a80071f5542992bc0c4a684/6` | Lara Croft and the Guardian of Light → game mode → single-player campaign | Прямо сказано «A single-player campaign mode is available». Связь поддерживается; qualifier про отсутствие AI относится к режиму, а не к наличию игры. |
| 3 | `H/5a72a2935542991f9a20c546/7` | Elana Amsterdam → website account on → elanaspantry.com | Источник говорит «Her blog, elanaspantry.com», но не об аккаунте на платформе. Нужна проверка смысла отношения после онтологического преобразования. |
| 4 | `H/5abbd3ac55429931dba1458b/9` | natural gas pipeline → instance of → natural gas pipeline | Самоссылка класса вместо факта об именованном Gulfstream Natural Gas Pipeline. Исправить субъект или исключить из обучающих целей. |
| 5 | `H/5ab849d955429916710eb017/9` | United States Marine Corps → has part(s) → United States Marine Corps | В тексте Marines — часть союзных сил, а не часть самих себя. Неподдерживаемая самоссылка. |
| 6 | `H/5a81a4615542995ce29dcc1e/7` | Mascarene teal → relative to → Bernier's teal | Источник использует «closest relative is probably». Категорический факт теряет модальность; qualifier про Madagascar её не восстанавливает. |
| 7 | `H/5a7ca0ef55429907fabeefd3/5` | Andrea Hlavackova → stage reached → No. 58 | Число и September 2012 подтверждаются, но текст говорит о рейтинге, не этапе турнира. Кроме того, Hlavackova и Hlaváčková — вариант написания, не доказательство выдуманной сущности. |
| 8 | `H/5add3b035542997545bbbcce/9` | The Gate → part of → Utopia | Связь с альбомом поддерживается «lead single from her ninth studio album». Дата 2017 присутствует; дополнительные поля не должны превращаться в отдельные несвязанные факты. |
| 9 | `H/5a79caad5542994f819ef09f/9` | Shoki Kasahara → member of sports team → Yomiuri Giants | Источник говорит «last played for», поэтому прошлое членство поддерживается, текущее — нет. Сохранить временной смысл при нормализации. |
| 10 | `H/5ac4f7e955429924173fb523/6` | Daniel Clifford → notable work → Midsummer House | Работа в ресторане и две звезды упомянуты, но «notable work» неоднозначно передаёт отношение человека к ресторану. Предпочтительно более точное отношение работы. |
| 11 | `M/2hop__6870_16335/16` | Bray et al. 2010 study → creator → Bray et al. | «A 2010 study by Bray et al.» поддерживает связь и год. Нормализованный субъект отсутствует как единая буквальная подстрока, что само по себе не ошибка. |
| 12 | `M/2hop__839126_472799/8` | 2018 Winter Olympics → participant → Albania | Перечень «Albania (2)» подтверждает участника и число при учёте заголовка статьи. Самостоятельный абзац — таблица без явно названного события; нужна политика использования заголовка. |
| 13 | `M/3hop1__46571_87694_64412/19` | Peter → iconically alludes to → violent reaction in Gethsemane during Jesus' arrest | Источник говорит «perhaps foreshadowing». Извлечение теряет предположительность и может путать персонажа с изображённым действием. Не принимать как безусловную связь. |
| 14 | `M/4hop3__373866_5189_38229_86687/1` | Germany → price → 100 billion deutschmarks | В тексте условное «would have paid ... if the Soviets demanded it», а не состоявшийся платёж или цена. Qualifier «stated by» недостаточен; семантическое искажение. |
| 15 | `M/3hop2__125104_90098_10557/0` | Latin language → develops from → Latin language | Текст различает Classical и Medieval Latin. Их сведение к одному имени создаёт ошибочную самоссылку; нужна более точная нормализация сущностей. |
| 16 | `M/3hop1__24980_42197_18397/18` | foreign nations → different from → foreign nations | Фраза об отказе других держав признавать японское правление не поддерживает это отношение. Удалить ошибочную цель. |
| 17 | `M/2hop__142699_67465/17` | Beyond the Beyond → instance of → role-playing video game | Жанр указан прямо; выбранный факт поддерживается текстом. |
| 18 | `M/2hop__25788_25839/14` | USB mice → compatible with → computers | Совместимость описана для старых компьютеров с PS/2 и при использовании адаптера. Удаление условия создаст чрезмерно общее утверждение; qualifier существенен. |
| 19 | `M/3hop2__88342_93066_47738/10` | Jose Miguel Cabrera Torres → award received → American League Most Valuable Player | «two-time ... MVP award winner» поддерживает связь и количество 2. Отсутствие акцента в Jose — сигнал нормализации, а не галлюцинация. |
| 20 | `M/4hop1__88342_75218_128008_80487/11` | LeBron James → award received → NBA Most Valuable Player Award | Четыре награды перечислены явно. Основная связь и выбранный количественный qualifier поддерживаются. |
| 21 | `G/3hop1__92903_53576_49385_Answerable/12` | Imperialism (term) → language of work or name → English | Введение термина в английский в late 1870s описано прямо. Субъект нормализован, но связь поддерживается; широкое онтологическое имя отношения требует единой инструкции. |
| 22 | `G/2hop__753314_23140_Unanswerable/13` | Fiscal year 2016 → total liabilities → 19,428 | Число взято из таблицы национального долга, но субъектом выбран год, не государство; утрачены единица billions и ограничение Oct. 2015–Jul. 2016. Нужен отдельный разбор табличных данных. |
| 23 | `G/3hop1__481899_839945_66869_Answerable/1` | A Man Called Dagger → content descriptor → low-budget | «low-budget spy film» поддерживает характеристику. Редкое отношение не обязательно является ошибочным. |
| 24 | `G/2hop__136565_123863_Answerable/7` | Season 3 (2017) → time in the pouch → 46 | В таблице 46 относится к Days in camp. Связь «time in the pouch» не передаёт этот смысл. Не обучать такое соответствие без исправления. |
| 25 | `G/2hop__643569_55607_Unanswerable/3` | Somebody's Watching Me → part of → Somebody's Watching Me | Это песня и одноимённый альбом, что видно из типов и текста. Совпадение имён допустимо; автоматическое удаление всех самоссылок потеряет корректную связь. |
| 26 | `G/2hop__767960_55607_Answerable/19` | Izzie Stevens → has effect → Izzie Stevens | Сюжет описывает заболевание и реанимацию, но не эту абстрактную самоссылку. Требуется исправление связи, а не только JSON. |
| 27 | `G/2hop__620475_24846_Unanswerable/2` | Security forces → has effect → demonstrators | В тексте конкретно «killed thousands of demonstrators». Отношение слишком общее и теряет существенный смысл действия. |
| 28 | `G/2hop__432945_51957_Unanswerable/11` | Charles Edward Herbert → position held → Member of the South Australian House of Assembly | Должность, годы 1900–1905 и округ указаны. Выбранный факт и qualifiers поддерживаются; при экспорте сохранить временную привязку. |
| 29 | `G/2hop__627916_89066_Answerable/13` | American Economic Journal → instance of → group of peer-reviewed academic journals | Начальная фраза прямо описывает группу из четырёх журналов. Основная связь поддерживается, но число 4 данным фактом не сохранено. |
| 30 | `G/2hop__559251_39289_Unanswerable/4` | Catherine Hayes Bailey → nominated by → National Peach Council | Источник говорит «was honored», не о номинации. Участники верны, но отношение искажает событие. |

## Следствия

1. Структурно валидный JSON не гарантирует правильность отношения.
2. Нельзя удалять все self-links: одноимённые сущности требуют различения, а не слияния.
3. Нормализация к онтологическим отношениям может менять смысл. Сравнение до/после
   обработки и аудит замен важнее одного количества сохранённых триплетов.
4. Условия, время, единицы измерения и модальность могут быть обязательной частью
   факта. Удаление типов и qualifiers требует семантической проверки.
5. Таблицы и абзацы с неявным субъектом требуют заголовка или более широкого контекста.
   Нельзя автоматически объявлять весь абзац корректным evidence каждого факта.
6. Эталон для оценки нужно размечать по тексту независимо от этих учительских целей.
