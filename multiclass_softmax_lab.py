#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проект 13 — Multiclass / Softmax Lab.

Три сырых логита превращаются в одно нормированное распределение:

    zA zB zC -> m -> z - m -> exp(z - m) -> сумма -> pA pB pC -> argmax

Главный объект интерфейса — симплекс: равносторонний треугольник, у которого
каждая точка внутри есть правильный ответ Softmax. Приложение показывает
механику нормировки, а не обучение и не геометрию классификатора: весов,
признаков, датасетов и границ решений здесь нет намеренно (это предметы
Classification Lab и проектов об обучении). Перекрёстная энтропия показана
как число выбранной цели — но только показана: по ней ничего не пересчитывается,
никакого градиентного спуска, эпох и обновления весов в приложении нет.

Все числа считаются по фактическим значениям: ничего не заложено заранее и
не подкрашивается «красивее, чем есть». Реализация Softmax сразу устойчивая —
с вычитанием максимального логита; наивная форма показывается отдельно, как
учебный контраст, и никогда не роняет приложение.

Только стандартная библиотека Python: tkinter, ttk, math. Без NumPy, внешних
библиотек, сети и файлов данных. Язык интерфейса переключается на ходу
(русский по умолчанию, английский), состояние расчёта при этом сохраняется.

Автор: SteveWorks001

Куда смотреть:
    Раздел 1   Константы, палитра, границы значений
    Раздел 2   Математика Softmax (без GUI)
    Раздел 3   Формулы и учебные пресеты
    Раздел 4   Локализация: TEXTS (ru / en)
    Раздел 5   Служебное форматирование чисел
    Раздел 6   Поле «слайдер + ввод»
    Раздел 7   Симплекс — главный график
    Раздел 8   Две шкалы: логиты против вероятностей
    Раздел 9   Стопка-полоса распределения и сравнение температур
    Раздел 10  Кривые чувствительности к одному логиту
    Раздел 11  Приложение
    Раздел 12  Точка входа
"""

import math
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

# ---------------------------------------------------------------------------
# Раздел 1. Константы, палитра, границы значений
# ---------------------------------------------------------------------------

AUTHOR = "SteveWorks001"

CLASS_KEYS = ("cls_a", "cls_b", "cls_c")
CLASS_TOKENS = ("zA", "zB", "zC")
CLASS_SHORT = ("A", "B", "C")
N_CLASSES = 3

DEFAULT_LOGITS = [2.0, 1.0, 0.0]
DEFAULT_TEMP = 1.0
DEFAULT_SHIFT = 0.0
DEFAULT_DELTA = 0.0

SLIDER_LO, SLIDER_HI = -12.0, 12.0
LOGIT_HARD = 100000.0
TEMP_MIN, TEMP_MAX = 0.1, 5.0
SHIFT_LO, SHIFT_HI = -60.0, 60.0
DELTA_LO, DELTA_HI = -8.0, 8.0

# Largest exponent representable in float64: math.exp(709.78) is finite,
# math.exp(710) already raises OverflowError.
EXP_LIMIT = 709.78

TEMP_FIXED = (0.1, 0.5, 1.0, 2.0, 5.0)
GRID_LEVELS = (0.25, 0.5, 0.75)
STEP_COUNT = 7
TRAIL_MAX = 18
# Measured rounding floor of the shift experiment: adding c to every logit
# changes the probabilities by at most ~1e-14 (T = 1) or ~2.5e-14 (T = 0.1),
# so anything under 1e-12 is "the same distribution, up to float64".
SHIFT_TOL = 1e-12
MAX_ENTROPY = math.log(N_CLASSES)

MIN_WIDTH = 1280
MIN_HEIGHT = 700
# Длины переносов подобраны по самой узкой колонке при окне 1280 px: подпись
# никогда не попросит себе ширину непереносной строки.
PAGE_WRAP = 400
CENTER_WRAP = 380
LEFT_WRAP = 248
HEAD_WRAP = 348

PAD_L = 10
COL_BG = "#fbfbf7"
COL_PAPER = "#fdfdf9"
COL_TEXT = "#242424"
COL_HEAD = "#31566f"
COL_KEY = "#1d3a4c"
COL_FRAME = "#c3ccd2"
COL_DIM = "#6d6d66"
COL_GRID = "#dfe3e2"
COL_AXIS = "#8d9aa2"
COL_A = "#2f6f8f"
COL_B = "#a9661b"
COL_C = "#3d7a4c"
COL_A_SOFT = "#cfe0ea"
COL_B_SOFT = "#f0dcc2"
COL_C_SOFT = "#cfe4d3"
COL_POINT = "#20262b"
COL_HI = "#fff3c4"
COL_BAD_BG = "#fde8e6"
COL_BAD_FG = "#7a1a13"
COL_GOOD = "#1d6b3f"
COL_RING = "#7a1f6d"
CLASS_COLORS = (COL_A, COL_B, COL_C)
CLASS_SOFTS = (COL_A_SOFT, COL_B_SOFT, COL_C_SOFT)

# ---------------------------------------------------------------------------
# Раздел 2. Математика Softmax (без GUI)
# ---------------------------------------------------------------------------


def clamp(value, lo, hi):
    """Значение внутри отрезка [lo, hi]."""
    return lo if value < lo else (hi if value > hi else value)


def scale_logits(logits, temperature):
    """Логиты, поделённые на температуру. T = 1 оставляет их без изменений.

    Температура обязана быть строго положительной: при T = 0 деления нет,
    а при T < 0 порядок классов переворачивается. Ноль и отрицательное число —
    это ошибка вызова, а не результат.
    """
    if temperature <= 0:
        raise ValueError("нужна температура строго больше нуля, получено %r"
                         % (temperature,))
    return [z / temperature for z in logits]


def softmax_chain(logits, temperature=1.0):
    """Устойчивый Softmax и все его промежуточные величины.

    Это единственный производственный путь расчёта: экспонента берётся от
    неположительного числа, поэтому переполнения не бывает ни при каких
    логитах, которые умеет принимать приложение.
    """
    scaled = scale_logits(logits, temperature)
    m = max(scaled)
    shifted = [v - m for v in scaled]
    exps = [math.exp(v) for v in shifted]
    denom = math.fsum(exps)
    probs = [e / denom for e in exps]
    return {
        "scaled": scaled,
        "max": m,
        "shifted": shifted,
        "exps": exps,
        "denom": denom,
        "probs": probs,
    }


def softmax(logits, temperature=1.0):
    """Только вероятности, без промежуточных шагов."""
    return softmax_chain(logits, temperature)["probs"]


def naive_chain(logits, temperature=1.0):
    """Наивный Softmax: exp берётся прямо от логита.

    Возвращает статус вместо исключения: приложение показывает, где наивный
    путь ломается, но не падает.
    """
    scaled = scale_logits(logits, temperature)
    exps = []
    flags = []
    for v in scaled:
        if v > EXP_LIMIT:
            exps.append(float("inf"))
            flags.append("overflow")
            continue
        try:
            e = math.exp(v)
        except OverflowError:
            e = float("inf")
        if e == 0.0:
            flags.append("underflow")
        else:
            flags.append("finite")
        exps.append(e)
    denom = math.fsum(exps)
    if any(f == "overflow" for f in flags) or math.isinf(denom):
        status = "overflow"
        probs = None
    elif denom == 0.0:
        status = "undefined"
        probs = None
    else:
        status = "finite"
        probs = [e / denom for e in exps]
    return {"scaled": scaled, "exps": exps, "flags": flags, "denom": denom,
            "probs": probs, "status": status}


def add_constant(logits, c):
    """Общий сдвиг: одно и то же число прибавляется ко всем логитам."""
    return [z + c for z in logits]


def perturb_one(logits, index, delta):
    """Меняется ровно один логит, остальные остаются как были."""
    out = list(logits)
    out[index] = out[index] + delta
    return out


def argmax_index(values):
    """Индекс наибольшего значения.

    При точной ничьей выигрывает меньший индекс: порядок правила A -> B -> C.
    Сравнение строгое (>, не >=), поэтому первый максимум и остаётся первым.
    """
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def is_tie(values, index):
    """Есть ли ещё одно значение, ровно такое же, как у выбранного."""
    return any(i != index and values[i] == values[index]
               for i in range(len(values)))


def entropy(probs):
    """H = -сумма p ln p. Нулевая вероятность даёт вклад 0, log(0) не берётся."""
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log(p)
    return h


def one_hot(target):
    """Вектор цели: единица у выбранного класса, ноль у двух других."""
    return [1.0 if k == target else 0.0 for k in range(N_CLASSES)]


def cross_entropy(chain, target):
    """Перекрёстная энтропия выбранного класса из уже посчитанной цепочки.

    Считается как log(S) - u, а не как -log(p): у крайне бедного класса
    вероятность равна нулю в float64, а потери при этом остаются конечным
    числом, потому что u и S взяты до деления.
    """
    return math.log(chain["denom"]) - chain["shifted"][target]


def probs_minus_target(chain, target):
    """Разность p - y по каждому классу: у цели неположительная, у прочих неотрицательная."""
    y = one_hot(target)
    return [chain["probs"][k] - y[k] for k in range(N_CLASSES)]


def max_abs_diff(p, q):
    """Самое большое расхождение двух распределений покомпонентно."""
    return max(abs(a - b) for a, b in zip(p, q))


def simplex_point(pA, pB, pC, vertices):
    """Барицентрические координаты -> точка на плоскости.

    Вершины идут в порядке A, B, C. Сумма весов равна 1, поэтому точка
    лежит внутри треугольника ровно тогда, когда все вероятности
    неотрицательны, — то есть всегда, как их посчитает Softmax.
    """
    ax, ay = vertices[0]
    bx, by = vertices[1]
    cx, cy = vertices[2]
    return (pA * ax + pB * bx + pC * cx, pA * ay + pB * by + pC * cy)


# ---------------------------------------------------------------------------
# Раздел 3. Формулы и учебные пресеты
# ---------------------------------------------------------------------------

F_STD = "pK = exp(zK) / (exp(zA) + exp(zB) + exp(zC))"
F_STABLE = "m = max(zA, zB, zC);   pK = exp(zK - m) / (exp(zA - m) + exp(zB - m) + exp(zC - m))"
F_TEMP = "pK = exp(zK / T) / (exp(zA / T) + exp(zB / T) + exp(zC / T))"
F_SHIFT = "Softmax(zA + c, zB + c, zC + c) = Softmax(zA, zB, zC)"
F_PRED = "pred = argmax(pA, pB, pC)"
F_ENTROPY = "H = -(pA ln pA + pB ln pB + pC ln pC),  0 <= H <= ln 3"
F_CE = "CE = -ln p(target) = ln S - u(target)"

PRESETS = (
    ("preset_1", (5.0, 1.0, -1.0)),
    ("preset_2", (2.0, 1.9, 1.8)),
    ("preset_3", (0.0, 0.0, 0.0)),
    ("preset_4", (-1.0, -2.0, -3.0)),
    ("preset_5", (300.0, 290.0, 280.0)),
)

STABILITY_CASES = (
    ("case_current", None),
    ("case_positive", (900.0, 850.0, 800.0)),
    ("case_negative", (-900.0, -950.0, -1000.0)),
    ("case_mixed", (710.0, 0.0, -710.0)),
)

STEP_CHIPS = ("chip_1", "chip_2", "chip_3", "chip_4", "chip_5", "chip_6", "chip_7")
STEP_TITLES = tuple("s%d_t" % i for i in range(1, STEP_COUNT + 1))
STEP_FORMULAS = (
    "zA, zB, zC",
    "m = max(zA / T, zB / T, zC / T)",
    "uK = zK / T - m",
    "eK = exp(uK)",
    "S = eA + eB + eC",
    "pK = eK / S",
    "pred = argmax(pA, pB, pC)",
)
STEP_EXPLAINS = tuple("s%d_e" % i for i in range(1, STEP_COUNT + 1))

HELP_ITEMS = tuple("h%d" % i for i in range(1, 17))

# ---------------------------------------------------------------------------
# Раздел 4. Локализация: TEXTS (ru / en)
# ---------------------------------------------------------------------------

LANG = "ru"


def set_language(code):
    global LANG
    LANG = code


def get_language():
    return LANG


TEXTS = {
    "ru": {
        "app_title": "Multiclass / Softmax Lab — лаборатория нормировки вероятностей",
        "subtitle": "Три логита -> одно распределение: что делает Softmax, если менять сами числа",
        "status_start": "Двигайте zA, zB, zC — точка в симплексе и все числа пересчитываются сразу",
        "status_lang": "Язык интерфейса: {name}",
        "status_preset": "Пресет: {name}",
        "status_reset": "Сброшено к zA=2.0, zB=1.0, zC=0.0 при T=1.0",
        "status_invalid": "Поле «{field}»: нужно число; осталось прежнее значение",
        "status_bad_temp": "Поле «{field}»: нужна температура строго больше нуля, получено {value}; осталось прежнее значение",
        "status_clamped": "Поле «{field}»: значение ограничено диапазоном, стало {value}",
        "status_offslider": "Поле «{field}»: значение вне диапазона ползунка, ползунок показывает границу",
        "status_step": "Шаг {n} из {total}: {title}",
        "status_temp": "T = {t}: {hint}",
        "status_shift": "Общий сдвиг c = {c}",
        "status_delta": "Меняется {name}: дельта = {d}",
        "status_ce": "Цель: {name}",
        "status_case": "Случай: {name}",
        "box_controls": "Три равноправных входа",
        "box_logits": "Логиты (сырые оценки)",
        "hint_logits": "Логит — любое число: отрицательное, нулевое, больше единицы. Вероятностью оно не является.",
        "hint_offslider": "вне диапазона ползунка",
        "box_temp": "Температура T",
        "name_temp": "Температура",
        "hint_temp": "T > 0. Делит логиты перед экспонентой: меняет резкость распределения, но не порядок классов.",
        "temp_sharp": "T < 1 — распределение острее: лидер забирает больше",
        "temp_standard": "T = 1 — обычный Softmax",
        "temp_flat": "T > 1 — распределение плоское, ближе к равновероятному",
        "box_presets": "Готовые наборы логитов",
        "preset_head": "Набор",
        "preset_1": "1. Явный лидер (5, 1, -1)",
        "preset_2": "2. Почти равные (2.0, 1.9, 1.8)",
        "preset_3": "3. Одинаковые (0, 0, 0)",
        "preset_4": "4. Все отрицательные (-1, -2, -3)",
        "preset_5": "5. Сотни (300, 290, 280)",
        "btn_reset_all": "Сброс",
        "box_simplex": "Симплекс: точка внутри — распределение",
        "simplex_hint": "Вершина = вся вероятность у одного класса, центр = ровно по трети каждому. Сумма координат всегда 1, поэтому точка не может выйти за треугольник.",
        "simplex_axis_note": "сетка: 25 / 50 / 75",
        "lbl_ring_shift": "кольцо: распределение после общего сдвига c",
        "lbl_ring_delta": "кольцо: распределение после дельты",
        "lbl_trail": "пунктир — путь точки за последние изменения",
        "box_dist": "Распределение",
        "lbl_prediction": "Прогноз (argmax)",
        "lbl_max_prob": "Максимальная вероятность",
        "lbl_entropy": "Энтропия H",
        "lbl_entropy_scale": "шкала 0 ... ln 3 = {v}",
        "lbl_sum": "Проверка суммы",
        "hint_dist": "Вероятности всегда положительны и в сумме дают 1: это нормированное значение, а не оценка величины.",
        "box_chain": "Цепочка расчёта (устойчивая форма)",
        "col_class": "Класс",
        "row_m": "максимум m",
        "row_sum": "знаменатель S",
        "chain_note": "Столбец z / T совпадает со столбцом z при T = 1. m вычитается перед экспонентой, поэтому экспоненты не больше 1.",
        "box_formulas": "Формулы",
        "ex_std": "Обычная запись Softmax: экспонента от каждого логита, поделённая на сумму таких экспонент.",
        "ex_stable": "Та же величина, но перед экспонентой вычитается максимум. Числа те же, переполнения нет.",
        "ex_temp": "Температура делит логиты: маленькая T усиливает различия, большая их сглаживает.",
        "ex_shift": "Один и тот же c у всех логитов сокращается в дроби — распределение не меняется.",
        "ex_pred": "Ответ модели — индекс наибольшей вероятности; порядок классов при равенстве фиксирован.",
        "ex_entropy": "Энтропия измеряет рассеянность распределения: 0 у сосредоточенного, ln 3 у равномерного.",
        "cls_a": "Класс A",
        "cls_b": "Класс B",
        "cls_c": "Класс C",
        "tab_zp": "z и p",
        "tab_temp": "T",
        "tab_shift": "Сдвиг",
        "tab_delta": "Один z",
        "tab_ce": "p и y",
        "tab_stable": "Устойч.",
        "tab_steps": "Шаги",
        "tab_help": "Справка",
        "zp_intro": "Две разные шкалы над одними и теми же числами: сверху логиты (любой знак и величина), снизу вероятности (отрезок от 0 до 1).",
        "zp_lane_logits": "логиты: шкала не ограничена",
        "zp_lane_probs": "вероятности: только 0 ... 1",
        "zp_note": "Пунктирные связки показывают, во что превращается каждый логит. Логит больше 1 или меньше 0 — нормально; вероятность такое значение дать не может.",
        "temp_intro": "Одних и тех же трёх логитов достаточно: меняется только T.",
        "temp_head_max": "argmax",
        "temp_head_ent": "H",
        "temp_tag_current": "текущая",
        "temp_note": "Каждый столбец — одна температура, три её вероятности; строка над числами показывает ту же полосу длиной 1. Порядок вероятностей при T > 0 сохраняется, меняется только резкость.",
        "shift_intro": "К каждому логиту прибавляется одно и то же число c.",
        "shift_lbl_c": "Общий сдвиг c",
        "shift_head_orig": "z",
        "shift_head_p": "p от z",
        "shift_head_shifted": "z + c",
        "shift_head_ps": "p от z + c",
        "shift_head_diff": "|разница|",
        "shift_verdict_same": "Распределения совпали с точностью плавающей точки: максимум расхождения {diff}. Общий множитель exp(c / T) сокращается в числителе и знаменателе.",
        "shift_verdict_diff": "Расхождения: {diff}. Проверьте, что сдвиг применён ко всем трём логитам.",
        "shift_expl": "Вычитать максимум — частный случай того же свойства: сдвиг на -m ничего не меняет, поэтому экспоненты остаются в безопасной области.",
        "delta_intro": "Меняется ровно один логит: выбран на один класс, величина — дельта.",
        "delta_lbl_class": "Какой логит двигать",
        "delta_lbl_d": "Дельта",
        "delta_head_before": "p до",
        "delta_head_after": "p после",
        "delta_head_diff": "сдвиг",
        "delta_axis_d": "дельта",
        "delta_axis_p": "вероятность",
        "delta_expl_coupled": "pA растёт, а pB и pC падают одновременно: знаменатель общий, поэтому вероятности связаны. Отдельной «своей» вероятности у класса нет.",
        "ce_intro": "Один из трёх классов объявляется правильным: у него в цели стоит 1, у двух других — 0. Перекрёстная энтропия измеряет, далеко ли распределение от такой цели.",
        "ce_lbl_target": "Какой класс считать целью",
        "ce_head_y": "цель y",
        "ce_head_p": "вероятность p",
        "ce_head_py": "разница p - y",
        "ce_row_lnS": "логарифм знаменателя: ln S",
        "ce_row_u": "сдвинутый логит цели",
        "ce_row_ce": "перекрёстная энтропия",
        "ce_verdict_match": "Цель совпала с argmax: {name} и есть выбранный класс. Потери малы ровно потому, что у цели большая вероятность.",
        "ce_verdict_miss": "Цель — {name}, а argmax выбрал {best}: распределение спорит с целью, поэтому потери больше нуля.",
        "ce_sum_line": "Сумма трёх разностей p - y равна {v}: это 0 с точностью до последнего знака, ведь вероятности дают 1 и цель тоже даёт 1.",
        "ce_expl_grad": "Знаки разностей не случайны: у выбранного класса p - y не больше нуля, у двух других — не меньше. Именно эти разности являются производными потерь по числам uK = zK / T; по самим логитам zK производная меньше в T раз. Здесь это только подпись: ничего не пересчитывается и не обучается.",
        "ce_expl_zero": "Если вероятность класса совсем мала, на экране стоит 0.000000, а число потерь остаётся конечным: оно взято из ln S и сдвинутого логита, а не из логарифма показанной вероятности. Нули в таблице — честный результат, а не сбой.",
        "chip_1": "логиты",
        "chip_2": "максимум",
        "chip_3": "сдвиг z-m",
        "chip_4": "экспоненты",
        "chip_5": "знаменатель",
        "chip_6": "вероятности",
        "chip_7": "argmax",
        "steps_intro": "Шаги используют текущие логиты и текущую T. Кнопки: «Далее», «Назад», «Сбросить»; работают стрелки влево и вправо.",
        "btn_prev": "Назад",
        "btn_next": "Далее",
        "btn_reset_steps": "Сбросить",
        "step_of": "Шаг {n} из {total}",
        "s1_t": "Логиты",
        "s1_e": "Три сырых оценки одного и того же объекта. Они не обязаны лежать в (0, 1) и не обязаны суммироваться в 1.",
        "s2_t": "Максимум",
        "s2_e": "Самый большой логит (после деления на T). Он нужен только для устойчивости: на него сдвигаются все три числа.",
        "s3_t": "Сдвинутые логиты",
        "s3_e": "Каждому логиту вычтен максимум, поэтому ни одно число не положительно. Наибольшее из них — ровно 0.",
        "s4_t": "Экспоненты",
        "s4_e": "Экспонента превращает разницу в отношение: прирост на 1 всегда умножает число на e, поэтому небольшое преимущество логита становится заметным преимуществом вероятности.",
        "s5_t": "Знаменатель",
        "s5_e": "Сумма трёх экспонент. Она положительна всегда: exp не даёт нулей и отрицательных значений.",
        "s6_t": "Вероятности",
        "s6_e": "Каждая экспонента делится на одну и ту же сумму — отсюда и 0 < p < 1, и сумма, равная 1.",
        "s7_t": "Argmax",
        "s7_e": "Выбирается класс с наибольшей вероятностью. Softmax при этом не обучается и не «знает», правильный ли класс.",
        "stable_intro": "Наивный путь берёт exp прямо от логита. Устойчивый — сначала вычитает максимум.",
        "stable_lbl_case": "Какие логиты взять",
        "case_current": "текущие логиты",
        "case_positive": "большие положительные (900, 850, 800)",
        "case_negative": "большие отрицательные (-900, -950, -1000)",
        "case_mixed": "смешанные (710, 0, -710)",
        "stable_head_z": "z",
        "stable_head_naive": "exp(z / T): наивно",
        "stable_head_safe": "exp(z / T - m): устойчиво",
        "stable_row_sum_naive": "S: наивно",
        "stable_row_sum_safe": "S: устойчиво",
        "stable_row_p_naive": "p: наивно",
        "stable_row_p_safe": "p: устойчиво",
        "stable_row_verdict_naive": "итог наивного пути",
        "stable_row_verdict_safe": "итог устойчивого пути",
        "st_finite": "конечно",
        "st_overflow": "переполнение -> inf",
        "st_underflow": "слишком мало -> 0",
        "st_undefined": "0 / 0, результата нет",
        "stable_note_limit": "граница float64: exp(x) конечен примерно до x = {v}; дальше — inf. Отрицательные x уходят к 0 (это underflow, не ошибка).",
        "stable_verdict_naive": "Наивный путь: {res}.",
        "stable_verdict_safe": "Устойчивый путь: {res}. Распределение то же там, где наивный путь ещё что-то считает.",
        "help_title": "Справка: как читать это приложение",
        "tie_rule": "Правило ничьей: при одинаковых максимальных вероятностях выбирается класс в порядке A, затем B, затем C. Правило детерминировано, случайности нет.",
        "calib_note": "Граница демонстрации: здесь показана арифметика и геометрия нормированного распределения. Приложение не определяет, откалиброваны ли вероятности и правильный ли выбранный класс.",
        "h1_t": "Что такое логит",
        "h1_b": "Логит — сырое число на выходе модели до любой нормировки. Его смысл только в сравнении с другими логитами.",
        "h2_t": "Почему логиты не вероятности",
        "h2_b": "Вероятность лежит между 0 и 1, и все вероятности одного исхода суммируются в 1. Логит свободен от обоих условий: он может быть -5 или 300.",
        "h3_t": "Знак и величина логита",
        "h3_b": "Отрицательный логит не значит «плохая вероятность»: после Softmax отрицательные логиты дают вполне приличные вероятности. Важен разрыв между числами, а не их положение относительно нуля.",
        "h4_t": "Что делает экспонента",
        "h4_b": "Экспонента переводит разницу в кратность: exp(a + 1) всегда в e раз больше exp(a). Поэтому разрыв в 1 единицу становится отношением примерно 2.7, а разрыв в 4 — отношением примерно 55.",
        "h5_t": "Почему вероятности суммируются в 1",
        "h5_b": "Каждое слагаемое делят на одну и ту же сумму всех слагаемых. Сложение даёт ровно 1 с точностью до последнего знака плавающей точки.",
        "h6_t": "Почему наибольший логит даёт наибольшую вероятность",
        "h6_b": "exp монотонна: большему логиту соответствует большая экспонента, а общий знаменатель только уменьшает все числа одинаково. Порядок сохраняется при любом T > 0.",
        "h7_t": "Почему один логит двигает все вероятности",
        "h7_b": "Знаменатель общий. Увеличивая один член, вы одновременно увеличиваете сумму, а значит уменьшаете долю двух других. На графике «Один z» это видно как две падающие кривые при растущей третьей.",
        "h8_t": "Что такое численная устойчивость",
        "h8_b": "float64 не умеет exp(x) примерно после x = 709: получается inf, а из inf и 0/0 нормировка не восстанавливается. Устойчивость — это приём, который не даёт таким числам появиться.",
        "h9_t": "Зачем вычитать максимум",
        "h9_b": "После вычитания наибольший логит становится нулём, остальные — отрицательными, и все экспоненты лежат в (0, 1]. Дробь не меняется, потому что числитель и знаменатель домножены на один и тот же exp(-m).",
        "h10_t": "Почему общий сдвиг ничего не меняет",
        "h10_b": "exp(z + c) = exp(z) * exp(c). Множитель exp(c) стоит в каждом слагаемом числителя и знаменателя и сокращается. Вкладка «Сдвиг» проверяет это числами: кольцо совпадает с точкой.",
        "h11_t": "Что делает температура",
        "h11_b": "T делит логиты перед экспонентой. T — не обучаемый параметр в этом приложении, а ручка, показывающая, как резко распределение наказывает проигравших.",
        "h12_t": "Низкая и высокая температура",
        "h12_b": "T меньше 1 растягивает различия: лидер почти забирает всё, энтропия падает. T больше 1 сглаживает: все трое похожи, энтропия растёт к ln 3. При T = 1 — обычный Softmax.",
        "h13_t": "Argmax и ничья",
        "h13_b": "При одинаковых максимальных вероятностях выбирается класс в порядке A, затем B, затем C: сравнение строгое, поэтому первый найденный максимум остаётся первым. Случайности в правиле нет.",
        "h14_t": "Чем этот проект не является",
        "h14_b": "Это не классификатор и не обучалка: нет признаков, весов, смещения, датасета, эпох, границы решений и accuracy. Перекрёстная энтропия показана как число, но по ней ничего не пересчитывается: производные здесь только названы в тексте, а не вычисляются.",
        "h15_t": "Чем проект отличается от Classification Lab",
        "h15_b": "Девятый проект отвечает на вопрос «где проходит граница и сколько точек ошиблись». Здесь вопроса о границе нет вовсе: на экран поданы готовые оценки, и разбирается только то, как из них получается распределение.",
        "h16_t": "Что такое перекрёстная энтропия",
        "h16_b": "Это одно число о паре «распределение и цель»: оно равно -ln p у выбранного класса и считается как ln S минус его сдвинутый логит. У совпадающего с целью уверенного класса число близко к 0, у спорящего с распределением — большое. Вкладка только показывает это число: потерю никто не минимизирует.",
    },
    "en": {
        "app_title": "Multiclass / Softmax Lab — a laboratory of probability normalization",
        "subtitle": "Three logits -> one distribution: what Softmax does when you edit the numbers themselves",
        "status_start": "Move zA, zB, zC — the simplex point and every number update at once",
        "status_lang": "Interface language: {name}",
        "status_preset": "Preset: {name}",
        "status_reset": "Reset to zA=2.0, zB=1.0, zC=0.0 at T=1.0",
        "status_invalid": "Field \"{field}\": a number is required; the previous value is kept",
        "status_bad_temp": "Field \"{field}\": a temperature strictly above zero is required, got {value}; the previous value is kept",
        "status_clamped": "Field \"{field}\": value limited to the range, now {value}",
        "status_offslider": "Field \"{field}\": value is outside the slider range, the slider shows its end",
        "status_step": "Step {n} of {total}: {title}",
        "status_temp": "T = {t}: {hint}",
        "status_shift": "Common shift c = {c}",
        "status_delta": "Editing {name}: delta = {d}",
        "status_ce": "Target: {name}",
        "status_case": "Case: {name}",
        "box_controls": "Three inputs of equal rank",
        "box_logits": "Logits (raw scores)",
        "hint_logits": "A logit is any number: negative, zero, larger than one. It is not a probability.",
        "hint_offslider": "outside the slider range",
        "box_temp": "Temperature T",
        "name_temp": "Temperature",
        "hint_temp": "T > 0. It divides the logits before exponentiation: it changes sharpness, never the class order.",
        "temp_sharp": "T < 1 — the distribution gets sharper: the leader takes more",
        "temp_standard": "T = 1 — standard Softmax",
        "temp_flat": "T > 1 — the distribution flattens toward the uniform one",
        "box_presets": "Ready sets of logits",
        "preset_head": "Set",
        "preset_1": "1. Clear leader (5, 1, -1)",
        "preset_2": "2. Almost equal (2.0, 1.9, 1.8)",
        "preset_3": "3. All equal (0, 0, 0)",
        "preset_4": "4. All negative (-1, -2, -3)",
        "preset_5": "5. Hundreds (300, 290, 280)",
        "btn_reset_all": "Reset",
        "box_simplex": "Simplex: every point inside is a valid distribution",
        "simplex_hint": "A vertex puts all the probability on one class, the center is exactly one third each. Coordinates always add up to 1, so the point cannot leave the triangle.",
        "simplex_axis_note": "gridlines: 25 / 50 / 75",
        "lbl_ring_shift": "ring: the distribution after the common shift c",
        "lbl_ring_delta": "ring: the distribution after the delta",
        "lbl_trail": "dashed line — where the point moved over recent changes",
        "box_dist": "Distribution",
        "lbl_prediction": "Prediction (argmax)",
        "lbl_max_prob": "Largest probability",
        "lbl_entropy": "Entropy H",
        "lbl_entropy_scale": "scale 0 ... ln 3 = {v}",
        "lbl_sum": "Sum check",
        "hint_dist": "Probabilities are always positive and add up to 1: this is a normalized value, not a measure of size.",
        "box_chain": "Calculation chain (stable form)",
        "col_class": "Class",
        "row_m": "maximum m",
        "row_sum": "denominator S",
        "chain_note": "The z / T column equals the z column at T = 1. The maximum is subtracted before exponentiation, so no exponential exceeds 1.",
        "box_formulas": "Formulas",
        "ex_std": "The usual way to write Softmax: the exponent of one logit over the sum of all three exponents.",
        "ex_stable": "The same value, but the maximum is subtracted first. The numbers match, the overflow does not happen.",
        "ex_temp": "Temperature divides the logits: a small T sharpens the gaps, a large T smooths them.",
        "ex_shift": "The same c added to every logit cancels in the fraction — the distribution stays as it was.",
        "ex_pred": "The answer is the index of the largest probability; ties follow a fixed class order.",
        "ex_entropy": "Entropy measures how spread the distribution is: 0 when concentrated, ln 3 when uniform.",
        "cls_a": "Class A",
        "cls_b": "Class B",
        "cls_c": "Class C",
        "tab_zp": "z vs p",
        "tab_temp": "T",
        "tab_shift": "Shift",
        "tab_delta": "One z",
        "tab_ce": "Cross-Entropy",
        "tab_stable": "Stable",
        "tab_steps": "Steps",
        "tab_help": "Help",
        "zp_intro": "Two different scales over the same numbers: logits on top (any sign, any size), probabilities below (the segment from 0 to 1).",
        "zp_lane_logits": "logits: unbounded scale",
        "zp_lane_probs": "probabilities: only 0 ... 1",
        "zp_note": "Each dashed link shows what a logit turns into. A logit above 1 or below 0 is normal; a probability can never take such a value.",
        "temp_intro": "The very same three logits are enough: only T changes.",
        "temp_head_max": "argmax",
        "temp_head_ent": "H",
        "temp_tag_current": "current",
        "temp_note": "The pA, pB, pC columns are one bar of length 1 cut differently. The order stays the same for any T > 0; only the sharpness changes.",
        "shift_intro": "The same number c is added to every logit.",
        "shift_lbl_c": "Common shift c",
        "shift_head_orig": "z",
        "shift_head_p": "p of z",
        "shift_head_shifted": "z + c",
        "shift_head_ps": "p of z + c",
        "shift_head_diff": "|difference|",
        "shift_verdict_same": "The distributions match to floating-point precision: largest gap {diff}. The common factor exp(c / T) cancels in the numerator and the denominator.",
        "shift_verdict_diff": "Gap: {diff}. Check that the shift was applied to all three logits.",
        "shift_expl": "Subtracting the maximum is a special case of the same property: a shift by -m changes nothing, which is why the exponentials stay in a safe range.",
        "delta_intro": "Exactly one logit changes: the class is picked, the size is the delta.",
        "delta_lbl_class": "Which logit to move",
        "delta_lbl_d": "Delta",
        "delta_head_before": "p before",
        "delta_head_after": "p after",
        "delta_head_diff": "change",
        "delta_axis_d": "delta",
        "delta_axis_p": "probability",
        "delta_expl_coupled": "pA grows while pB and pC fall at the same time: the denominator is shared, so the probabilities are coupled. A class has no probability of its own.",
        "ce_intro": "One of the three classes is declared correct: the target holds 1 for it and 0 for the other two. Cross-entropy measures how far the distribution is from such a target.",
        "ce_lbl_target": "Which class is the target",
        "ce_head_y": "target y",
        "ce_head_p": "probability p",
        "ce_head_py": "difference p - y",
        "ce_row_lnS": "log of the denominator: ln S",
        "ce_row_u": "shifted logit of the target",
        "ce_row_ce": "cross-entropy",
        "ce_verdict_match": "The target matches argmax: {name} is exactly the chosen class. The loss is small precisely because the target carries a large probability.",
        "ce_verdict_miss": "The target is {name}, argmax chose {best}: the distribution disagrees with the target, so the loss is above zero.",
        "ce_sum_line": "The three differences p - y add up to {v}: that is 0 up to the last digit, since the probabilities sum to 1 and the target sums to 1 as well.",
        "ce_expl_grad": "The signs are not accidental: the chosen class has p - y at or below zero and the other two have it at or above zero. Those same differences are the derivatives of the loss with respect to uK = zK / T; with respect to the logits zK they are smaller by a factor of T. Here that is only a caption: nothing is recomputed and nothing is trained.",
        "ce_expl_zero": "If a class probability is tiny, the screen shows 0.000000 while the loss stays finite: it comes from ln S and the shifted logit, not from the logarithm of the displayed probability. Zeros in the table are an honest result, not a failure.",
        "chip_1": "logits",
        "chip_2": "maximum",
        "chip_3": "shifted z-m",
        "chip_4": "exponentials",
        "chip_5": "denominator",
        "chip_6": "probabilities",
        "chip_7": "argmax",
        "steps_intro": "Steps use the current logits and the current T. Buttons: Next, Previous, Reset; the left and right arrows work too.",
        "btn_prev": "Previous",
        "btn_next": "Next",
        "btn_reset_steps": "Reset",
        "step_of": "Step {n} of {total}",
        "s1_t": "Logits",
        "s1_e": "Three raw scores of one and the same object. They do not have to lie in (0, 1) and do not have to add up to 1.",
        "s2_t": "Maximum",
        "s2_e": "The largest logit (after the division by T). It exists only for stability: all three numbers are shifted by it.",
        "s3_t": "Shifted logits",
        "s3_e": "The maximum is subtracted from every logit, so no number is positive. The largest of them is exactly 0.",
        "s4_t": "Exponentials",
        "s4_e": "Exponentiation turns a gap into a ratio: a growth by 1 always multiplies the value by e, so a small lead in logits becomes a visible lead in probability.",
        "s5_t": "Denominator",
        "s5_e": "The sum of the three exponentials. It is always positive: exp gives neither zeros nor negatives.",
        "s6_t": "Probabilities",
        "s6_e": "Each exponential is divided by that same sum — which is where 0 < p < 1 and the sum of 1 come from.",
        "s7_t": "Argmax",
        "s7_e": "The class with the largest probability is picked. Softmax does not train and does not know whether the class is right.",
        "stable_intro": "The naive path takes exp straight from the logit. The stable one subtracts the maximum first.",
        "stable_lbl_case": "Which logits to take",
        "case_current": "the current logits",
        "case_positive": "large positive (900, 850, 800)",
        "case_negative": "large negative (-900, -950, -1000)",
        "case_mixed": "mixed (710, 0, -710)",
        "stable_head_z": "z",
        "stable_head_naive": "exp(z / T): naive",
        "stable_head_safe": "exp(z / T - m): stable",
        "stable_row_sum_naive": "S: naive",
        "stable_row_sum_safe": "S: stable",
        "stable_row_p_naive": "p: naive",
        "stable_row_p_safe": "p: stable",
        "stable_row_verdict_naive": "naive verdict",
        "stable_row_verdict_safe": "stable verdict",
        "st_finite": "finite",
        "st_overflow": "overflow -> inf",
        "st_underflow": "too small -> 0",
        "st_undefined": "0 / 0, no result",
        "stable_note_limit": "float64 limit: exp(x) stays finite roughly up to x = {v}; beyond it comes inf. Large negative x slide to 0 (that is underflow, not an error).",
        "stable_verdict_naive": "Naive path: {res}.",
        "stable_verdict_safe": "Stable path: {res}. Wherever the naive path still computes something, the distribution is the same.",
        "help_title": "Help: how to read this application",
        "tie_rule": "Tie rule: when several probabilities are exactly the maximum, the class is chosen in the order A, then B, then C. The rule is deterministic, there is no randomness.",
        "calib_note": "Boundary of the demonstration: this shows the arithmetic and the geometry of a normalized distribution. It does not decide whether the probabilities are calibrated or whether the chosen class is actually correct.",
        "h1_t": "What a logit is",
        "h1_b": "A logit is a raw number coming out of a model before any normalization. It only means something next to the other logits.",
        "h2_t": "Why logits are not probabilities",
        "h2_b": "A probability sits between 0 and 1 and all the probabilities of one outcome add to 1. A logit obeys neither rule: it can be -5 or 300.",
        "h3_t": "Sign and size of a logit",
        "h3_b": "A negative logit does not mean a bad probability: after Softmax negative logits give perfectly decent probabilities. What matters is the gap between the numbers, not their position relative to zero.",
        "h4_t": "What exponentiation does",
        "h4_b": "Exponentiation converts a difference into a multiple: exp(a + 1) is always e times exp(a). So a gap of 1 becomes a ratio of about 2.7 and a gap of 4 a ratio of about 55.",
        "h5_t": "Why the probabilities sum to 1",
        "h5_b": "Every term is divided by the sum of all the terms. Adding them gives exactly 1 up to the last floating-point digit.",
        "h6_t": "Why the largest logit gives the largest probability",
        "h6_b": "exp is monotonic: a larger logit means a larger exponential, and the shared denominator shrinks every number the same way. The order survives for any T > 0.",
        "h7_t": "Why one logit moves all the probabilities",
        "h7_b": "The denominator is shared. Raising one term raises the whole sum, which shrinks the share of the other two. On the One z tab this is two falling curves while the third rises.",
        "h8_t": "What numerical stability means",
        "h8_b": "float64 cannot do exp(x) past roughly x = 709: it yields inf, and normalization cannot be recovered from inf or from 0/0. Stability is the trick that keeps such numbers from appearing.",
        "h9_t": "Why the maximum is subtracted",
        "h9_b": "After the subtraction the largest logit is 0 and the rest are negative, so every exponential lies in (0, 1]. The fraction does not change because numerator and denominator are both multiplied by exp(-m).",
        "h10_t": "Why a common shift changes nothing",
        "h10_b": "exp(z + c) = exp(z) * exp(c). The factor exp(c) stands in every term of the numerator and the denominator and cancels. The Shift tab checks it numerically: the ring lands on the point.",
        "h11_t": "What temperature does",
        "h11_b": "T divides the logits before exponentiation. Here T is not a learned parameter but a handle that shows how harshly the distribution punishes the losers.",
        "h12_t": "Low and high temperature",
        "h12_b": "T below 1 stretches the gaps: the leader nearly takes everything and entropy falls. T above 1 smooths them: the three look alike and entropy grows toward ln 3. At T = 1 it is the standard Softmax.",
        "h13_t": "Argmax and ties",
        "h13_b": "When several probabilities are exactly the maximum, the class is taken in the order A, then B, then C: the comparison is strict, so the first maximum found stays first. The rule holds no randomness.",
        "h14_t": "What this project is not",
        "h14_b": "It is not a classifier and not a training lab: there are no features, weights, bias, dataset, epochs, decision boundary or accuracy. Cross-entropy is shown as a number, but nothing is recomputed from it: the derivatives are only named in the text, not calculated.",
        "h15_t": "How it differs from the Classification Lab",
        "h15_b": "Project 9 answers where the boundary passes and how many points are misclassified. There is no boundary question at all here: raw scores are handed to the screen and only the way a distribution is made out of them is examined.",
        "h16_t": "What cross-entropy is",
        "h16_b": "One number about a pair «distribution and target»: it equals -ln p for the chosen class and is computed as ln S minus that class's shifted logit. A confident class that matches the target gives a value close to 0, a class the distribution argues with gives a large one. The tab only shows the number: nothing minimizes the loss.",
    },
}


def T(key, **kwargs):
    """Доступ к локализованному тексту с подстановкой."""
    table = TEXTS.get(get_language()) or TEXTS["ru"]
    if key not in table:
        table = TEXTS["ru"]
    if key not in table:
        return "[" + key + "]"
    text = table[key]
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass
    return text


# ---------------------------------------------------------------------------
# Раздел 5. Служебное форматирование чисел
# ---------------------------------------------------------------------------


def fmt(value, digits=4):
    """Обычный вывод числа; inf, nan и None читаются как текст, а не как сбой."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "inf"
    if abs(value) >= 1e5:
        return "%.6g" % value
    return "%.*f" % (digits, value)


def fmt_prob(value):
    """Вероятность: шесть знаков, чтобы сумма 1 читалась без сюрпризов."""
    return fmt(value, 6)


def fmt_tiny(value):
    """Мелкая вероятность или экспонента: крошечное и огромное — степенью."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if value == 0.0:
        return "0.000000"
    magnitude = abs(value)
    if magnitude < 1e-4 or magnitude >= 1e4:
        return "%.2e" % value
    return "%.6f" % value


def fmt_precise(value):
    """Полная точность для сравнения двух распределений."""
    if value is None:
        return "-"
    if value == 0.0:
        return "0.000000000000"
    if abs(value) < 1e-12:
        return "%.3e" % value
    return "%.12f" % value


def fmt_gap(value):
    """Расхождение двух вероятностей: тот же порядок, но влезает в колонку."""
    if value is None:
        return "-"
    if value == 0.0:
        return "0.000000"
    if abs(value) < 1e-4:
        return "%.3e" % value
    return "%.6f" % value


def fmt_signed(value):
    """Знаковое число из отрезка [-1, 1]: разность p - y, не длиннее девяти знаков."""
    if value == 0.0:
        return "+0.000000"
    if abs(value) < 1e-4:
        return "%+.1e" % value
    return "%+.6f" % value


def fmt_compact(value):
    """Вероятность для таблицы температур: пять знаков, мелкое — степенью."""
    if value is None:
        return "-"
    if value >= 0.001:
        return "%.3f" % value
    if value == 0.0:
        return "0.000"
    return "%.0e" % value


def fmt_chain(value):
    """Промежуточное число таблицы шагов: не длиннее восьми знаков."""
    if value is None:
        return "-"
    magnitude = abs(value)
    if magnitude >= 1000.0 or (magnitude != 0.0 and magnitude < 1e-4):
        return "%.1e" % value
    return "%.3f" % value


def fmt_input(value, digits):
    return "%.*f" % (digits, value)


def parse_number(text):
    """Число из поля ввода: запятая равна точке, мусор возвращает None."""
    raw = (text or "").strip().replace(",", ".")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def tick_values(lo, hi, count=5):
    """Ровные деления шкалы."""
    if hi <= lo:
        return [lo]
    step = (hi - lo) / float(count)
    return [lo + step * i for i in range(count + 1)]


def nice_limit(value):
    """Круглая граница симметричной шкалы."""
    if value <= 1.0:
        return 1.0
    for base in (1.0, 2.0, 2.5, 5.0, 10.0):
        for power in range(0, 8):
            cand = base * (10.0 ** power)
            if cand >= value:
                return cand
    return value


# ---------------------------------------------------------------------------
# Раздел 6. Поле «слайдер + ввод»
# ---------------------------------------------------------------------------


class ValueField(ttk.Frame):
    """Одно число: подпись, слайдер и поле ввода связаны, ввод проверяется."""

    def __init__(self, master, app, name_key, token, color, lo, hi, digits,
                 value, on_change=None, use_slider=True, validate=None):
        ttk.Frame.__init__(self, master)
        self.app = app
        self.name_key = name_key
        self.token = token
        self.lo = lo
        self.hi = hi
        self.digits = digits
        self.on_change = on_change
        self.validate = validate
        self.value = value
        self._silent = False

        self.swatch = tk.Canvas(self, width=11, height=11, highlightthickness=0,
                                bg=COL_PAPER)
        self.swatch.create_oval(1, 1, 10, 10, fill=color, outline=color)
        self.swatch.pack(side="left", padx=(2, 6))

        self.lbl_name = ttk.Label(self, style="Key.TLabel")
        self.lbl_name.pack(side="left")
        self.retranslate()

        if use_slider:
            self.var_scale = tk.DoubleVar(value=clamp(value, lo, hi))
            self.scale = ttk.Scale(self, from_=lo, to=hi, orient="horizontal",
                                   variable=self.var_scale,
                                   command=self._on_scale)
            self.scale.pack(side="left", fill="x", expand=True, padx=(8, 8))
        else:
            self.scale = None

        self.entry_var = tk.StringVar(value=fmt_input(value, digits))
        self.entry = ttk.Entry(self, textvariable=self.entry_var, width=9,
                               justify="right")
        self.entry.pack(side="right")
        self.entry.bind("<Return>", self._commit)
        self.entry.bind("<FocusOut>", self._commit)

        self.lbl_hint = ttk.Label(self, style="Tiny.TLabel")

    def retranslate(self):
        self.lbl_name.configure(text="%s  %s" % (T(self.name_key), self.token))

    def _on_scale(self, _raw):
        if self._silent:
            return
        value = round(self.var_scale.get(), self.digits)
        self.value = value
        self._silent = True
        self.entry_var.set(fmt_input(value, self.digits))
        self._silent = False
        self._mark(False)
        self._hide_hint()
        if self.on_change is not None:
            self.on_change(value)

    def _commit(self, _event=None):
        if self._silent:
            return
        parsed = parse_number(self.entry_var.get())
        if parsed is None:
            self._reject("status_invalid", field=T(self.name_key))
            return
        if self.validate is not None:
            bad_key = self.validate(parsed)
            if bad_key is not None:
                self._reject(bad_key, field=T(self.name_key),
                             value=fmt(parsed, self.digits))
                return
        value = parsed
        if abs(value) > LOGIT_HARD:
            value = clamp(value, -LOGIT_HARD, LOGIT_HARD)
            self.app.set_status("status_clamped", field=T(self.name_key),
                                value=fmt(value, 1))
        elif not (self.lo <= value <= self.hi):
            self.app.set_status("status_offslider", field=T(self.name_key))
        self.set_value(value, notify=True)

    def set_value(self, value, notify=False):
        self.value = value
        self._silent = True
        if self.scale is not None:
            self.var_scale.set(clamp(value, self.lo, self.hi))
        self.entry_var.set(fmt_input(value, self.digits))
        self._silent = False
        self._mark(False)
        self._show_hint(value)
        if notify and self.on_change is not None:
            self.on_change(value)

    def _reject(self, status_key, **kwargs):
        """Отклонённый ввод: поле краснеет, в нём остаётся прежнее значение."""
        self._mark(True)
        self.app.set_status(status_key, **kwargs)
        self._silent = True
        self.entry_var.set(fmt_input(self.value, self.digits))
        self._silent = False

    def _show_hint(self, value):
        if self.scale is None or self.lo <= value <= self.hi:
            self._hide_hint()
            return
        self.lbl_hint.configure(text=T("hint_offslider"), foreground=COL_DIM)
        # первым в списке pack: тогда заметка занимает всю ширину строки,
        # а не узкую щель между ползунком и полем ввода
        self.lbl_hint.pack(side="bottom", anchor="w", before=self.swatch,
                          pady=(1, 0))

    def _hide_hint(self):
        if self.lbl_hint.winfo_manager():
            self.lbl_hint.pack_forget()

    def _mark(self, bad):
        self.entry.configure(style="BadEntry.TEntry" if bad else "GoodEntry.TEntry")


# ---------------------------------------------------------------------------
# Раздел 7. Симплекс — главный график
# ---------------------------------------------------------------------------


class SimplexView(tk.Canvas):
    """Равносторонний треугольник вероятностей: барицентрические координаты.

    Точка строится как pA * A + pB * B + pC * C, поэтому она внутри
    треугольника ровно тогда, когда все три вероятности неотрицательны.
    """

    def __init__(self, master, app, height=336):
        tk.Canvas.__init__(self, master, height=height, bg=COL_PAPER,
                           highlightthickness=0, bd=0)
        self.app = app
        self.height = height
        self.bind("<Configure>", lambda _event: self.redraw())

    def vertices(self, w, h):
        margin_x = 34
        margin_top = 44
        margin_bottom = 54
        avail_h = max(20.0, h - margin_top - margin_bottom)
        avail_w = max(20.0, w - 2 * margin_x)
        side = min(avail_w, avail_h / (math.sqrt(3.0) / 2.0))
        cx = w / 2.0
        top = margin_top
        bottom = margin_top + side * math.sqrt(3.0) / 2.0
        return ((cx, top), (cx - side / 2.0, bottom), (cx + side / 2.0, bottom))

    def _chord(self, verts, k, level):
        """Концы линии, на которой вероятность класса k равна level."""
        ends = []
        for j in (i for i in range(N_CLASSES) if i != k):
            probs = [0.0, 0.0, 0.0]
            probs[k] = level
            probs[j] = 1.0 - level
            ends.append(simplex_point(probs[0], probs[1], probs[2], verts))
        return ends[0], ends[1]

    def redraw(self):
        self.delete("all")
        w = max(60, self.winfo_width())
        h = max(60, self.height if self.winfo_height() < 2 else self.winfo_height())
        verts = self.vertices(w, h)
        tiny = self.app.fonts["tiny"]
        smallb = self.app.fonts["smallb"]

        for k in range(N_CLASSES):
            for level in GRID_LEVELS:
                a, b = self._chord(verts, k, level)
                self.create_line(a[0], a[1], b[0], b[1], fill=CLASS_SOFTS[k],
                                 width=1)
                mx = (a[0] + b[0]) / 2.0
                my = (a[1] + b[1]) / 2.0
                # короткие метки: две соседние семейства сетки не сливаются
                label = "%d" % round(level * 100)
                self.create_rectangle(mx - 9, my - 6, mx + 9, my + 6,
                                      fill=COL_PAPER, outline=COL_PAPER)
                self.create_text(mx, my, text=label, font=tiny,
                                 fill=CLASS_COLORS[k])

        self.create_polygon(verts[0][0], verts[0][1], verts[1][0], verts[1][1],
                            verts[2][0], verts[2][1], fill="", outline=COL_HEAD,
                            width=2)

        corners = ((0, -16, "s"), (1, 14, "n"), (2, 14, "n"))
        for k, dy, anchor in corners:
            vx, vy = verts[k]
            dx = 0 if k == 0 else (-8 if k == 1 else 8)
            self.create_text(vx + dx, vy + dy, anchor=anchor,
                             text="%s: p = 1" % CLASS_SHORT[k], font=smallb,
                             fill=CLASS_COLORS[k])

        cx = sum(v[0] for v in verts) / 3.0
        cy = sum(v[1] for v in verts) / 3.0
        self.create_line(cx, cy, verts[0][0], verts[0][1], fill=COL_GRID)
        self.create_line(cx, cy, verts[1][0], verts[1][1], fill=COL_GRID)
        self.create_line(cx, cy, verts[2][0], verts[2][1], fill=COL_GRID)
        self.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill=COL_AXIS,
                         outline=COL_AXIS)

        probs = self.app.probs
        px, py = simplex_point(probs[0], probs[1], probs[2], verts)

        trail = self.app.trail
        if trail:
            pts = []
            for old in trail[-TRAIL_MAX:]:
                pts.extend(simplex_point(old[0], old[1], old[2], verts))
            pts.extend([px, py])
            if len(pts) >= 4:
                self.create_line(*pts, fill=COL_AXIS, dash=(2, 3), width=1)

        ring = self.app.ring
        if ring is not None:
            rx, ry = simplex_point(ring[0], ring[1], ring[2], verts)
            self.create_oval(rx - 10, ry - 10, rx + 10, ry + 10, outline=COL_RING,
                             width=2)

        point_label = (CLASS_SHORT[self.app.best] + " = " +
                       fmt_prob(probs[self.app.best]))
        self.create_oval(px - 7, py - 7, px + 7, py + 7, fill=COL_POINT,
                         outline=COL_PAPER, width=2)
        half = smallb.measure(point_label) / 2.0
        lx = px + 12
        if lx + 2.0 * half > w - 6:
            lx = px - 12 - 2.0 * half
        cx_text = min(max(lx + half, half + 4), w - half - 4)
        # у верхней вершины подпись ставим ниже точки, иначе она сливается
        # с надписью вершины; если мешаетет сетка - уходим ещё на шаг
        ly = py + 22 if py < verts[0][1] + 34 else py - 12
        step = 14 if ly > py else -14
        for _attempt in range(4):
            left_x = cx_text - half - 2
            right_x = cx_text + half + 2
            clash = False
            for item in self.find_all():
                if self.type(item) != "text":
                    continue
                ib = self.bbox(item)
                if ib and ib[0] < right_x and ib[2] > left_x \
                        and ib[1] < ly + 8 and ib[3] > ly - 8:
                    clash = True
                    break
            if not clash:
                break
            ly = min(max(ly + step, 12), h - 12)
        self.create_rectangle(cx_text - half - 2, ly - 8, cx_text + half + 2,
                              ly + 8, fill=COL_PAPER, outline=COL_PAPER)
        self.create_text(cx_text, ly, text=point_label, font=smallb,
                         fill=COL_POINT)

        self.create_text(w / 2.0, h - 16, font=tiny, fill=COL_DIM,
                         text=T("simplex_axis_note"))


# ---------------------------------------------------------------------------
# Раздел 8. Две шкалы: логиты против вероятностей
# ---------------------------------------------------------------------------


class ScaleView(tk.Canvas):
    """Две независимые шкалы над одним и тем же состоянием.

    Верхняя — сырые логиты: симметричная, без границ по знаку. Нижняя —
    вероятности: отрезок 0 ... 1. Пунктир показывает, во что превратился
    каждый логит; совпадения по горизонтали тут нет по построению.
    """

    def __init__(self, master, app, height=196):
        tk.Canvas.__init__(self, master, height=height, bg=COL_PAPER,
                           highlightthickness=0, bd=0)
        self.app = app
        self.height = height
        self.bind("<Configure>", lambda _event: self.redraw())

    def redraw(self):
        self.delete("all")
        w = max(80, self.winfo_width())
        h = max(60, self.height if self.winfo_height() < 2 else self.winfo_height())
        tiny = self.app.fonts["tiny"]
        smallb = self.app.fonts["smallb"]
        left = 58
        right = max(left + 40, w - 20)
        # Строки фиксированные: значения каждого класса всегда в своей полосе,
        # поэтому подписи не наезжают друг на друга при близких логитах.
        lane_logit = 40
        lane_prob = 118
        row_z = 74
        row_p = 152
        cap_p = min(h - 10, 176)

        logits = self.app.logits
        probs = self.app.probs

        limit = nice_limit(max(abs(v) for v in logits)) if logits else 1.0
        span = right - left

        def z_x(value):
            return left + (value + limit) / (2.0 * limit) * span

        def p_x(value):
            return left + value * span

        def fit(x, text, font):
            """Центр подписи, который не вылезает за край холста."""
            half = font.measure(text) / 2.0
            return min(max(x, half + 3), w - half - 3)

        def masked(x, y, text, font, fill, anchor="c"):
            """Текст на бумаге: пунктирные связи не мешают его читать."""
            half = font.measure(text) / 2.0
            self.create_rectangle(x - half - 2, y - 8, x + half + 2, y + 8,
                                  fill=COL_PAPER, outline=COL_PAPER)
            self.create_text(x, y, text=text, font=font, fill=fill,
                             anchor=anchor)

        self.create_text(4, 12, anchor="w", font=tiny, fill=COL_DIM,
                         text=T("zp_lane_logits"))
        self.create_text(w - 6, 12, anchor="e", font=tiny, fill=COL_DIM,
                         text="[-%s ... +%s]" % (fmt(limit, 0), fmt(limit, 0)))

        for k in range(N_CLASSES):
            zx = z_x(logits[k])
            px = p_x(probs[k])
            self.create_line(zx, lane_logit + 6, zx, lane_prob - 6,
                             fill=CLASS_COLORS[k], dash=(2, 3), width=1)

        self.create_line(left, lane_logit, right, lane_logit, fill=COL_AXIS,
                         arrow="last", width=1)
        for tick in tick_values(-limit, limit, 4):
            x = z_x(tick)
            self.create_line(x, lane_logit - 4, x, lane_logit + 4, fill=COL_AXIS)
            label = fmt(tick, 0)
            self.create_text(fit(x, label, tiny), lane_logit + 14, font=tiny,
                             fill=COL_DIM, text=label)
        zero_x = z_x(0.0)
        self.create_line(zero_x, lane_logit - 9, zero_x, lane_logit + 9,
                         fill=COL_HEAD, width=2)

        self.create_line(left, lane_prob, right, lane_prob, fill=COL_AXIS,
                         arrow="last", width=1)
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = p_x(tick)
            self.create_line(x, lane_prob - 4, x, lane_prob + 4, fill=COL_AXIS)
            label = "%.2f" % tick
            self.create_text(fit(x, label, tiny), lane_prob + 14, font=tiny,
                             fill=COL_DIM, text=label)
        self.create_line(left, lane_prob - 13, left, lane_prob + 13,
                         fill=COL_HEAD, width=2)

        slot = span / float(N_CLASSES)
        for k in range(N_CLASSES):
            zx = z_x(logits[k])
            px = p_x(probs[k])
            self.create_oval(zx - 5, lane_logit - 5, zx + 5, lane_logit + 5,
                             fill=CLASS_COLORS[k], outline=CLASS_COLORS[k])
            self.create_oval(px - 5, lane_prob - 5, px + 5, lane_prob + 5,
                             fill=CLASS_COLORS[k], outline=COL_PAPER, width=1)
            masked(left + slot * (k + 0.5), row_z,
                   "%s = %s" % (CLASS_TOKENS[k], fmt_chain(logits[k])),
                   smallb, CLASS_COLORS[k])
            masked(left + slot * (k + 0.5), row_p,
                   "p%s = %s" % (CLASS_SHORT[k], fmt_prob(probs[k])),
                   smallb, COL_TEXT)

        self.create_text(4, cap_p, anchor="w", font=tiny, fill=COL_DIM,
                         text=T("zp_lane_probs"))


# ---------------------------------------------------------------------------
# Раздел 9. Стопка-полоса распределения и сравнение температур
# ---------------------------------------------------------------------------


class StackBar(tk.Canvas):
    """Одна полоса длиной ровно 1, разрезанная на три части.

    Полоса заполнена от левого края до правого всегда: это и есть
    «pA + pB + pC = 1» в виде геометрии, а не подписи.
    """

    def __init__(self, master, app, height=56):
        tk.Canvas.__init__(self, master, height=height, bg=COL_PAPER,
                           highlightthickness=0, bd=0)
        self.app = app
        self.height = height
        self.bind("<Configure>", lambda _event: self.redraw())

    def redraw(self):
        self.delete("all")
        w = max(60, self.winfo_width())
        h = max(30, self.height if self.winfo_height() < 2 else self.winfo_height())
        x0, x1 = 22, w - 16
        top, bottom = 8, min(26, h - 24)
        span = float(x1 - x0)
        probs = self.app.probs
        tiny = self.app.fonts["tiny"]
        smallb = self.app.fonts["smallb"]

        x = x0
        for k in range(N_CLASSES):
            x_next = x0 + span * math.fsum(probs[:k + 1])
            self.create_rectangle(x, top, x_next, bottom,
                                  fill=CLASS_COLORS[k], outline=COL_PAPER,
                                  width=1)
            width_px = x_next - x
            if width_px > 52:
                label = "%s  %s" % (CLASS_SHORT[k], fmt_prob(probs[k])[:4])
                self.create_text((x + x_next) / 2.0, (top + bottom) / 2.0,
                                 text=label, font=smallb, fill="#ffffff")
            elif width_px > 3:
                self.create_text(x_next, bottom + 4, anchor="n", font=tiny,
                                 fill=CLASS_COLORS[k], text=CLASS_SHORT[k])
            x = x_next

        self.create_text(x0 - 8, (top + bottom) / 2.0, anchor="e", font=tiny,
                         fill=COL_DIM, text="0")
        self.create_text(x1 + 6, (top + bottom) / 2.0, anchor="w", font=tiny,
                         fill=COL_DIM, text="1")
        self.create_line(x1, top - 4, x1, bottom + 4, fill=COL_HEAD)
        self.create_line(x0, top - 4, x0, bottom + 4, fill=COL_HEAD)


class EntropyBar(tk.Canvas):
    """Короткая полоска 0 ... ln 3 для энтропии текущего распределения."""

    def __init__(self, master, width=132, height=16):
        tk.Canvas.__init__(self, master, width=width, height=height,
                           bg=COL_PAPER, highlightthickness=0, bd=0)
        self.bar_width = width

    def redraw(self, value):
        self.delete("all")
        span = self.bar_width - 2
        ratio = clamp(value / MAX_ENTROPY, 0.0, 1.0) if MAX_ENTROPY else 0.0
        self.create_rectangle(1, 4, 1 + span, 13, fill=COL_A_SOFT,
                              outline=COL_FRAME)
        self.create_rectangle(1, 4, 1 + span * ratio, 13, fill=COL_HEAD,
                              outline=COL_HEAD)
        mid = 1 + span * 0.5
        self.create_line(mid, 5, mid, 12, fill=COL_PAPER)


class TempCompare(ttk.Frame):
    """Одни и те же логиты при нескольких значениях T: столбцы — температуры."""

    BAR_WIDTH = 50

    def __init__(self, master, app):
        ttk.Frame.__init__(self, master)
        self.app = app
        self.cols = len(TEMP_FIXED) + 1
        self.heads = []
        self.bars = []
        self.prob_rows = []
        for _k in range(N_CLASSES):
            self.prob_rows.append([])
        self.argmax_cells = []
        self.ent_cells = []

        corner = ttk.Label(self, style="Head.TLabel", text="T")
        corner.grid(row=1, column=0, sticky="w")
        for k in range(N_CLASSES):
            name = ttk.Label(self, style="MonoBold.TLabel", anchor="w")
            name.configure(text=CLASS_TOKENS[k], foreground=CLASS_COLORS[k])
            name.grid(row=2 + k, column=0, sticky="w")
        for row, key in ((5, "temp_head_max"), (6, "temp_head_ent")):
            name = ttk.Label(self, style="Head.TLabel")
            name.grid(row=row, column=0, sticky="w")
            app.register(name, key)

        for col in range(self.cols):
            pad = {"padx": (0, 4), "sticky": "e"}
            head = ttk.Label(self, style="Head.TLabel", width=5, anchor="e")
            head.grid(row=0, column=col + 1, **pad)
            self.heads.append(head)
            bar = tk.Canvas(self, width=self.BAR_WIDTH, height=13,
                            bg=COL_PAPER, highlightthickness=0, bd=0)
            bar.grid(row=1, column=col + 1, **pad)
            self.bars.append(bar)
            for k in range(N_CLASSES):
                value = ttk.Label(self, style="Mono.TLabel", width=5, anchor="e")
                value.grid(row=2 + k, column=col + 1, **pad)
                self.prob_rows[k].append(value)
            best = ttk.Label(self, style="Mono.TLabel", width=5, anchor="e")
            best.grid(row=5, column=col + 1, **pad)
            self.argmax_cells.append(best)
            ent = ttk.Label(self, style="Mono.TLabel", width=5, anchor="e")
            ent.grid(row=6, column=col + 1, **pad)
            self.ent_cells.append(ent)

    def _column_widgets(self, col):
        return ([self.heads[col], self.bars[col]] +
                [self.prob_rows[k][col] for k in range(N_CLASSES)] +
                [self.argmax_cells[col], self.ent_cells[col]])

    def refresh(self):
        logits = self.app.logits
        temps = list(TEMP_FIXED)
        current = self.app.temperature
        if current not in temps:
            temps.append(current)
        for col in range(self.cols):
            widgets = self._column_widgets(col)
            if col >= len(temps):
                for widget in widgets:
                    widget.grid_remove()
                continue
            value = temps[col]
            is_current = abs(value - current) < 1e-9
            probs = softmax(logits, value)
            self.heads[col].configure(
                text="%.1f*" % value if is_current else "%.1f" % value,
                foreground=COL_POINT if is_current else COL_HEAD)
            x = 0.0
            self.bars[col].delete("all")
            for k in range(N_CLASSES):
                x_next = self.BAR_WIDTH * math.fsum(probs[:k + 1])
                self.bars[col].create_rectangle(x, 1, x_next, 12,
                                                fill=CLASS_COLORS[k],
                                                outline=COL_PAPER, width=1)
                x = x_next
            for k in range(N_CLASSES):
                self.prob_rows[k][col].configure(text=fmt_compact(probs[k]))
            self.argmax_cells[col].configure(text=CLASS_SHORT[argmax_index(probs)])
            self.ent_cells[col].configure(text=fmt_compact(entropy(probs)))
            for widget in widgets:
                widget.grid()


# ---------------------------------------------------------------------------
# Раздел 10. Кривые чувствительности к одному логиту
# ---------------------------------------------------------------------------


class DeltaCurves(tk.Canvas):
    """Три вероятности как функция от дельты одного логита."""

    def __init__(self, master, app, height=196):
        tk.Canvas.__init__(self, master, height=height, bg=COL_PAPER,
                           highlightthickness=0, bd=0)
        self.app = app
        self.height = height
        self.samples = 81
        self.bind("<Configure>", lambda _event: self.redraw())

    def redraw(self):
        self.delete("all")
        w = max(80, self.winfo_width())
        h = max(80, self.height if self.winfo_height() < 2 else self.winfo_height())
        tiny = self.app.fonts["tiny"]
        left, right = 52, w - 12
        top, bottom = 14, h - 48
        index = self.app.delta_class

        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = bottom - (bottom - top) * level
            self.create_line(left, y, right, y, fill=COL_GRID)
            self.create_text(left - 5, y, anchor="e", font=tiny, fill=COL_DIM,
                             text="%.2f" % level)
        for pos, d in enumerate((DELTA_LO, DELTA_LO / 2.0, 0.0,
                                 DELTA_HI / 2.0, DELTA_HI)):
            x = left + (right - left) * (d - DELTA_LO) / (DELTA_HI - DELTA_LO)
            self.create_line(x, top, x, bottom, fill=COL_GRID)
            # крайние деления уводим внутрь: иначе они лезут на подпись оси Y
            anchor = "w" if pos == 0 else ("e" if pos == 4 else "c")
            self.create_text(x, bottom + 12, font=tiny, fill=COL_DIM,
                             anchor=anchor, text=fmt(d, 1))

        def d_x(value):
            return left + (right - left) * (value - DELTA_LO) / (DELTA_HI - DELTA_LO)

        step = (DELTA_HI - DELTA_LO) / float(self.samples - 1)
        for k in range(N_CLASSES):
            pts = []
            for i in range(self.samples):
                d = DELTA_LO + step * i
                probs = softmax(perturb_one(self.app.logits, index, d),
                                self.app.temperature)
                pts.append(d_x(d))
                pts.append(bottom - (bottom - top) * probs[k])
            self.create_line(*pts, fill=CLASS_COLORS[k], width=2)

        now = d_x(self.app.delta)
        self.create_line(now, top, now, bottom, fill=COL_POINT, dash=(3, 2))
        probs = self.app.probs
        for k in range(N_CLASSES):
            y = bottom - (bottom - top) * probs[k]
            self.create_oval(now - 4, y - 4, now + 4, y + 4,
                             fill=CLASS_COLORS[k], outline=COL_PAPER)

        self.create_text((left + right) / 2.0, h - 14, font=tiny, fill=COL_DIM,
                         text=T("delta_axis_d"))
        self.create_text(10, (top + bottom) / 2.0, angle=270, font=tiny,
                         fill=COL_DIM, text=T("delta_axis_p"))


# ---------------------------------------------------------------------------
# Раздел 11. Приложение
# ---------------------------------------------------------------------------


class ScrollFrame(ttk.Frame):
    """Колонка с прокруткой: содержимое не обрезается на низких окнах."""

    def __init__(self, master, bg=COL_PAPER, border=True):
        ttk.Frame.__init__(self, master)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self, style="Card.TFrame" if border else "Body.TFrame")
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner,
                                                  anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind("<Enter>", self._enter)
        self.canvas.bind("<Leave>", self._leave)

    def _on_inner(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _enter(self, _event):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _leave(self, _event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _wheel(self, event):
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-2, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(2, "units")
        elif getattr(event, "delta", 0) > 0:
            self.canvas.yview_scroll(-2, "units")
        else:
            self.canvas.yview_scroll(2, "units")

    def home(self):
        self.canvas.yview_moveto(0.0)


class App(tk.Tk):
    """Три логита, одно распределение: всё считается на лету."""

    def __init__(self):
        tk.Tk.__init__(self)
        self._building = True
        self._reg = []
        self._build_fonts()
        self._build_styles()
        self._init_state()
        self.title(T("app_title"))
        self.configure(bg=COL_BG)
        self.geometry("1320x790")
        self.minsize(MIN_WIDTH, MIN_HEIGHT)
        self._build_header()
        self._build_body()
        self._build_statusbar()
        self.bind("<Left>", self._on_arrow)
        self.bind("<Right>", self._on_arrow)
        self._building = False
        self.set_status("status_start")
        self.refresh()
        self.after(80, self._zoom)

    def _zoom(self):
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

    # --- сборка шрифтов и стилей -------------------------------------------

    def _build_fonts(self):
        base = tkfont.nametofont("TkDefaultFont")
        fixed = tkfont.nametofont("TkFixedFont")
        self.fonts = {
            "title": tkfont.Font(font=base, size=13, weight="bold"),
            "sub": tkfont.Font(font=base, size=9),
            "body": tkfont.Font(font=base, size=10),
            "small": tkfont.Font(font=base, size=9),
            "smallb": tkfont.Font(font=base, size=9, weight="bold"),
            "tiny": tkfont.Font(font=base, size=8),
            "tinyb": tkfont.Font(font=base, size=8, weight="bold"),
            "mono": tkfont.Font(font=fixed, size=9),
            "monob": tkfont.Font(font=fixed, size=9, weight="bold"),
        }

    def _build_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        f = self.fonts
        style.configure(".", background=COL_BG, foreground=COL_TEXT,
                        font=f["body"])
        style.configure("Body.TFrame", background=COL_BG)
        style.configure("Card.TFrame", background=COL_PAPER)
        style.configure("Edge.TFrame", background=COL_PAPER, relief="solid",
                        bordercolor=COL_FRAME, lightcolor=COL_FRAME,
                        darkcolor=COL_FRAME, borderwidth=1)
        style.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        style.configure("Title.TLabel", background=COL_BG, foreground=COL_KEY,
                        font=f["title"])
        style.configure("Sub.TLabel", background=COL_BG, foreground=COL_DIM,
                        font=f["sub"])
        style.configure("CardHead.TLabel", background=COL_PAPER,
                        foreground=COL_HEAD, font=f["smallb"])
        style.configure("Key.TLabel", background=COL_PAPER, foreground=COL_TEXT,
                        font=f["small"])
        style.configure("Head.TLabel", background=COL_PAPER, foreground=COL_HEAD,
                        font=f["smallb"])
        style.configure("Tiny.TLabel", background=COL_PAPER, foreground=COL_DIM,
                        font=f["tiny"])
        style.configure("Mono.TLabel", background=COL_PAPER, foreground=COL_TEXT,
                        font=f["mono"])
        style.configure("MonoKey.TLabel", background=COL_PAPER,
                        foreground=COL_DIM, font=f["mono"])
        style.configure("MonoBold.TLabel", background=COL_PAPER,
                        foreground=COL_KEY, font=f["monob"])
        style.configure("Dim.TLabel", background=COL_PAPER, foreground=COL_FRAME,
                        font=f["mono"])
        style.configure("Chip.TLabel", background=COL_PAPER, foreground=COL_DIM,
                        font=f["tiny"])
        style.configure("ChipDim.TLabel", background=COL_PAPER,
                        foreground=COL_FRAME, font=f["tiny"])
        style.configure("ChipHi.TLabel", background=COL_HI, foreground=COL_KEY,
                        font=f["tinyb"])
        style.configure("Hi.TLabel", background=COL_HI, foreground=COL_KEY,
                        font=f["mono"])
        style.configure("Status.TLabel", background=COL_BG, foreground=COL_TEXT,
                        font=f["small"])
        style.configure("Author.TLabel", background=COL_BG, foreground=COL_DIM,
                        font=f["tiny"])
        style.configure("TButton", font=f["small"], padding=(8, 4))
        style.map("TButton",
                  background=[("active", COL_A_SOFT), ("pressed", COL_A_SOFT)])
        style.configure("Tool.TButton", font=f["smallb"], padding=(6, 2))
        style.configure("Lang.TButton", font=f["smallb"], padding=(6, 2),
                        foreground=COL_DIM)
        style.configure("LangActive.TButton", font=f["smallb"], padding=(6, 2),
                        foreground=COL_KEY, background=COL_A_SOFT,
                        lightcolor=COL_A_SOFT, darkcolor=COL_A_SOFT,
                        bordercolor=COL_HEAD)
        style.configure("TRadiobutton", background=COL_PAPER, font=f["small"],
                        focuscolor=COL_PAPER)
        for index in range(N_CLASSES):
            style.configure("Radio%d.TRadiobutton" % index,
                            background=COL_PAPER, foreground=CLASS_COLORS[index],
                            font=f["small"], focuscolor=COL_PAPER)
        style.configure("TNotebook", background=COL_BG, borderwidth=0,
                        tabmargins=(2, 4, 2, 0))
        style.configure("TNotebook.Tab", background=COL_PAPER,
                        foreground=COL_HEAD, font=f["smallb"], padding=(4, 5))
        style.map("TNotebook.Tab",
                  background=[("selected", COL_A_SOFT)],
                  foreground=[("selected", COL_KEY)])
        style.configure("TCombobox", fieldbackground=COL_PAPER,
                        background=COL_PAPER, arrowcolor=COL_TEXT, font=f["small"])
        style.configure("Case.TCombobox", fieldbackground=COL_PAPER,
                        background=COL_PAPER, arrowcolor=COL_TEXT,
                        foreground=COL_TEXT, font=f["small"])
        style.configure("Horizontal.TScale", background=COL_PAPER,
                        troughcolor=COL_A_SOFT)
        style.configure("TScrollbar", background=COL_PAPER, troughcolor=COL_BG,
                        arrowcolor=COL_DIM, bordercolor=COL_FRAME,
                        lightcolor=COL_FRAME, darkcolor=COL_FRAME)
        style.configure("GoodEntry.TEntry", fieldbackground=COL_PAPER,
                        foreground=COL_TEXT)
        style.configure("BadEntry.TEntry", fieldbackground=COL_BAD_BG,
                        foreground=COL_BAD_FG)
        style.map("GoodEntry.TEntry", bordercolor=[("focus", COL_HEAD)],
                  lightcolor=[("focus", COL_HEAD)])

    # --- состояние ----------------------------------------------------------

    def _init_state(self):
        self.logits = list(DEFAULT_LOGITS)
        self.temperature = DEFAULT_TEMP
        self.shift = DEFAULT_SHIFT
        self.delta = DEFAULT_DELTA
        self.delta_class = 0
        self.ce_target = 0
        self.stab_case = 0
        self.step = STEP_COUNT
        self.active_tab = "zp"
        self.chain = softmax_chain(self.logits, self.temperature)
        self.probs = list(self.chain["probs"])
        self.trail = []
        self.ring = None
        self.ring_key = None
        self.best = argmax_index(self.probs)

    # --- локализация --------------------------------------------------------

    def register(self, widget, key, **kwargs):
        """Подпись, которая перерисовывается при смене языка."""
        self._reg.append((widget, key, kwargs))
        widget.configure(text=T(key, **kwargs))
        return widget

    def set_status(self, key, **kwargs):
        if hasattr(self, "lbl_status"):
            self.lbl_status.configure(text=T(key, **kwargs))

    def switch_language(self, code):
        if code not in TEXTS:
            return
        set_language(code)
        self.title(T("app_title"))
        self._mark_lang_buttons()
        for widget, key, kwargs in self._reg:
            widget.configure(text=T(key, **kwargs))
        for widget in self._all_fields():
            widget.retranslate()
        for child, key in self._tab_refs:
            self.nb.tab(child, text=T(key))
        self.refresh()
        self.set_status("status_lang",
                        name="Русский" if code == "ru" else "English")

    def _all_fields(self):
        return list(self.logit_fields) + [self.temp_field, self.shift_field,
                                          self.delta_field]

    # --- карточки -----------------------------------------------------------

    def card(self, parent, key=None, padding=(8, 6)):
        outer = ttk.Frame(parent, style="Edge.TFrame")
        body = ttk.Frame(outer, style="Card.TFrame")
        if key is not None:
            head = ttk.Label(outer, style="CardHead.TLabel",
                             wraplength=HEAD_WRAP, justify="left")
            head.pack(fill="x", padx=padding[0], pady=(padding[1], 0))
            self.register(head, key)
        body.pack(fill="both", expand=True, padx=padding[0],
                  pady=(0, padding[1]))
        return outer, body

    # --- верх окна ----------------------------------------------------------

    def _build_header(self):
        head = ttk.Frame(self, style="Body.TFrame")
        head.pack(fill="x", padx=10, pady=(8, 2))
        head.grid_columnconfigure(0, weight=1)
        title = ttk.Label(head, style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w")
        self.register(title, "app_title")
        sub = ttk.Label(head, style="Sub.TLabel")
        sub.grid(row=1, column=0, sticky="w")
        self.register(sub, "subtitle")

        langs = ttk.Frame(head, style="Body.TFrame")
        langs.grid(row=0, column=1, rowspan=2, sticky="e")
        self.lang_buttons = {}
        for code in ("ru", "en"):
            btn = ttk.Button(langs, text=code.upper(), width=4, style="Tool.TButton",
                             command=lambda c=code: self.switch_language(c))
            btn.pack(side="left", padx=(4, 0))
            self.lang_buttons[code] = btn
        self._mark_lang_buttons()

    def _mark_lang_buttons(self):
        for code, btn in self.lang_buttons.items():
            btn.configure(style="LangActive.TButton" if code == get_language()
                          else "Lang.TButton")

    # --- тело ---------------------------------------------------------------

    def _build_body(self):
        body = ttk.Frame(self, style="Body.TFrame")
        body.pack(fill="both", expand=True, padx=8, pady=(2, 2))
        # все три колонки с прокруткой: на низком окне содержимое уезжает
        # за нижний край, но не обрезается и не сжимается в кашу
        body.grid_columnconfigure(0, minsize=300)
        body.grid_columnconfigure(1, weight=1, minsize=430)
        body.grid_columnconfigure(2, minsize=452)
        body.grid_rowconfigure(0, weight=1)

        self.col_left = ScrollFrame(body, COL_PAPER)
        self.col_left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.col_center = ScrollFrame(body, COL_BG, border=False)
        self.col_center.grid(row=0, column=1, sticky="nsew", padx=4)
        self.nb = ttk.Notebook(body)
        self.nb.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)

        self._build_controls_column()
        self._build_center_column()
        self._build_notebook()

    # --- левая колонка: входы, пресеты, формулы ------------------------------

    def _build_controls_column(self):
        parent = self.col_left.inner
        card, inner = self.card(parent, "box_controls")
        card.pack(fill="x", padx=2, pady=(0, 6))

        sub = ttk.Label(inner, style="Head.TLabel")
        sub.pack(anchor="w", pady=(2, 2))
        self.register(sub, "box_logits")

        self.logit_fields = []
        for k in range(N_CLASSES):
            field = ValueField(inner, self, CLASS_KEYS[k], CLASS_TOKENS[k],
                               CLASS_COLORS[k], SLIDER_LO, SLIDER_HI, 2,
                               self.logits[k],
                               on_change=lambda v, i=k: self._on_logit(i, v))
            field.pack(fill="x", pady=1)
            self.logit_fields.append(field)

        hint = ttk.Label(inner, style="Tiny.TLabel", wraplength=LEFT_WRAP,
                         justify="left")
        hint.pack(anchor="w", pady=(4, 0))
        self.register(hint, "hint_logits")

        sub2 = ttk.Label(inner, style="Head.TLabel")
        sub2.pack(anchor="w", pady=(8, 2))
        self.register(sub2, "box_temp")
        self.temp_field = ValueField(inner, self, "name_temp", "T", COL_HEAD,
                                     TEMP_MIN, TEMP_MAX, 2, self.temperature,
                                     on_change=self._on_temp,
                                     validate=self._temp_is_positive)
        self.temp_field.pack(fill="x")
        hint2 = ttk.Label(inner, style="Tiny.TLabel", wraplength=LEFT_WRAP,
                          justify="left")
        hint2.pack(anchor="w", pady=(4, 0))
        self.register(hint2, "hint_temp")

        card, inner = self.card(parent, "box_presets")
        card.pack(fill="x", padx=2, pady=(0, 6))
        caption = ttk.Label(inner, style="Tiny.TLabel")
        caption.pack(anchor="w")
        self.register(caption, "preset_head")
        self.preset_buttons = []
        for index, (key, _values) in enumerate(PRESETS):
            btn = ttk.Button(inner, command=lambda i=index:
                             self.apply_preset(i))
            btn.pack(fill="x", pady=1)
            self.register(btn, key)
            self.preset_buttons.append(btn)
        reset = ttk.Button(inner, text="", style="Tool.TButton",
                           command=self.reset_all)
        reset.pack(fill="x", pady=(6, 0))
        self.register(reset, "btn_reset_all")

        card, inner = self.card(parent, "box_formulas")
        card.pack(fill="x", padx=2, pady=(0, 6))
        pairs = ((F_STD, "ex_std"), (F_STABLE, "ex_stable"), (F_TEMP, "ex_temp"),
                 (F_SHIFT, "ex_shift"), (F_PRED, "ex_pred"),
                 (F_ENTROPY, "ex_entropy"))
        for formula, key in pairs:
            lbl_f = ttk.Label(inner, style="MonoBold.TLabel", wraplength=LEFT_WRAP,
                              justify="left")
            lbl_f.configure(text=formula)
            lbl_f.pack(anchor="w", pady=(5, 0))
            lbl_e = ttk.Label(inner, style="Tiny.TLabel", wraplength=LEFT_WRAP,
                              justify="left")
            lbl_e.pack(anchor="w")
            self.register(lbl_e, key)

    # --- центр: симплекс, распределение, цепочка -----------------------------

    def _build_center_column(self):
        parent = self.col_center.inner
        card, inner = self.card(parent, "box_simplex")
        card.pack(fill="x", pady=(0, 6))

        self.simplex = SimplexView(inner, self, height=252)
        self.simplex.pack(fill="x")

        legend = ttk.Frame(inner, style="Card.TFrame")
        legend.pack(fill="x", pady=(2, 0))
        self.lbl_trail = ttk.Label(legend, style="Tiny.TLabel",
                                   wraplength=CENTER_WRAP, justify="left")
        self.lbl_trail.pack(anchor="w")
        self.register(self.lbl_trail, "lbl_trail")
        self.lbl_ring = ttk.Label(legend, style="Tiny.TLabel",
                                  wraplength=CENTER_WRAP, justify="left",
                                  foreground=COL_RING)
        self.lbl_ring.pack(anchor="w")

        self.lbl_simplex_hint = ttk.Label(inner, style="Tiny.TLabel",
                                          wraplength=CENTER_WRAP, justify="left")
        self.lbl_simplex_hint.pack(anchor="w", pady=(2, 0))
        self.register(self.lbl_simplex_hint, "simplex_hint")

        card, inner = self.card(parent, "box_dist")
        card.pack(fill="x", pady=(0, 6))
        self.stack = StackBar(inner, self)
        self.stack.pack(fill="x")

        readouts = ttk.Frame(inner, style="Card.TFrame")
        readouts.pack(fill="x", pady=(6, 0))
        for col in range(3):
            readouts.grid_columnconfigure(col, weight=1)
        readout_keys = ("lbl_prediction", "lbl_max_prob", "lbl_sum")
        self.dist_values = []
        for col, key in enumerate(readout_keys):
            cell = ttk.Frame(readouts, style="Card.TFrame")
            cell.grid(row=0, column=col, sticky="w")
            lbl = ttk.Label(cell, style="Tiny.TLabel", wraplength=104,
                            justify="left")
            lbl.pack(anchor="w")
            self.register(lbl, key)
            value = ttk.Label(cell, style="MonoBold.TLabel")
            value.pack(anchor="w")
            self.dist_values.append(value)

        ent_row = ttk.Frame(inner, style="Card.TFrame")
        ent_row.pack(fill="x", pady=(6, 0))
        lbl_ent = ttk.Label(ent_row, style="Tiny.TLabel")
        lbl_ent.pack(side="left")
        self.register(lbl_ent, "lbl_entropy")
        self.lbl_ent_value = ttk.Label(ent_row, style="MonoBold.TLabel")
        self.lbl_ent_value.pack(side="left", padx=(6, 8))
        self.ent_bar = EntropyBar(ent_row)
        self.ent_bar.pack(side="left")
        scale = ttk.Label(inner, style="Tiny.TLabel", wraplength=CENTER_WRAP,
                          justify="left")
        scale.pack(anchor="e", pady=(1, 0))
        self.lbl_scale = scale
        self.register(scale, "lbl_entropy_scale", v=fmt(MAX_ENTROPY, 6))

        self.lbl_tie = ttk.Label(inner, style="Tiny.TLabel", wraplength=CENTER_WRAP,
                                 justify="left", foreground=COL_HEAD)
        self.lbl_tie.pack(anchor="w", pady=(4, 0))
        self.register(self.lbl_tie, "tie_rule")

        hint = ttk.Label(inner, style="Tiny.TLabel", wraplength=CENTER_WRAP,
                         justify="left")
        hint.pack(anchor="w", pady=(2, 0))
        self.register(hint, "hint_dist")

        card, inner = self.card(parent, "box_chain")
        card.pack(fill="x")
        grid = ttk.Frame(inner, style="Card.TFrame")
        grid.pack(fill="x")
        grid.grid_columnconfigure(0, minsize=126)
        for col in (1, 2, 3):
            grid.grid_columnconfigure(col, minsize=78)

        head = ttk.Label(grid, style="Head.TLabel")
        head.grid(row=0, column=0, sticky="w", pady=(0, 3))
        self.register(head, "col_class")
        for k in range(N_CLASSES):
            lbl = ttk.Label(grid, style="Head.TLabel", foreground=CLASS_COLORS[k])
            lbl.configure(text=CLASS_TOKENS[k])
            lbl.grid(row=0, column=k + 1, sticky="e", pady=(0, 3))

        rows = (("z", "logits", 1), ("z / T", "scaled", 2),
                (None, "m", 2), ("u = z/T - m", "shifted", 3),
                ("exp(u)", "exps", 4), (None, "denom", 5),
                ("p", "probs", 6), (None, "pred", 7))
        scalar_kinds = ("m", "denom", "pred")
        self.chain_rows = []
        for r, (token, kind, stage) in enumerate(rows, start=1):
            cells = {"stage": stage, "kind": kind, "labels": []}
            if token is not None:
                lbl = ttk.Label(grid, style="MonoKey.TLabel")
                lbl.configure(text=token)
            elif kind == "pred":
                lbl = ttk.Label(grid, style="MonoKey.TLabel")
                lbl.configure(text="pred")
            else:
                lbl = ttk.Label(grid, style="Key.TLabel")
                self.register(lbl, "row_m" if kind == "m" else "row_sum")
            lbl.grid(row=r, column=0, sticky="w", pady=1)
            cells["name"] = lbl
            if kind in scalar_kinds:
                val = ttk.Label(grid, style="Mono.TLabel", anchor="e")
                val.grid(row=r, column=1, columnspan=3, sticky="e", pady=1)
                cells["labels"].append(val)
            else:
                for k in range(N_CLASSES):
                    val = ttk.Label(grid, style="Mono.TLabel", anchor="e")
                    val.grid(row=r, column=k + 1, sticky="e", pady=1)
                    cells["labels"].append(val)
            self.chain_rows.append(cells)

        note = ttk.Label(inner, style="Tiny.TLabel", wraplength=CENTER_WRAP,
                         justify="left")
        note.pack(anchor="w", pady=(6, 0))
        self.register(note, "chain_note")

    # --- правая колонка: эксперименты ---------------------------------------

    def _page(self, parent):
        scroll = ScrollFrame(parent, COL_PAPER)
        scroll.pack(fill="both", expand=True)
        return scroll.inner

    def _label(self, parent, key, style="Key.TLabel", wrap=PAGE_WRAP,
               pady=(4, 2)):
        lbl = ttk.Label(parent, style=style, wraplength=wrap, justify="left")
        lbl.pack(anchor="w", pady=pady)
        self.register(lbl, key)
        return lbl

    def _build_notebook(self):
        self._pages = {}
        self._tab_refs = []
        self._tab_tags = []
        pages = (("tab_zp", "zp", self._page_zp),
                 ("tab_temp", "temp", self._page_temp),
                 ("tab_shift", "shift", self._page_shift),
                 ("tab_delta", "delta", self._page_delta),
                 ("tab_ce", "ce", self._page_ce),
                 ("tab_stable", "stable", self._page_stable),
                 ("tab_steps", "steps", self._page_steps),
                 ("tab_help", "help", self._page_help))
        for key, tag, builder in pages:
            frame = ttk.Frame(self.nb, style="Body.TFrame")
            self.nb.add(frame, text=T(key))
            self._pages[tag] = frame
            self._tab_refs.append((frame, key))
            self._tab_tags.append(tag)
            builder(frame)

    def _page_zp(self, parent):
        inner = self._page(parent)
        self._label(inner, "zp_intro", pady=(8, 4))
        self.scale_view = ScaleView(inner, self, height=196)
        self.scale_view.pack(fill="x", pady=(0, 4))
        self._label(inner, "zp_note", style="Tiny.TLabel", pady=(0, 8))

    def _page_temp(self, parent):
        inner = self._page(parent)
        self._label(inner, "temp_intro", pady=(8, 4))
        self.temp_compare = TempCompare(inner, self)
        self.temp_compare.pack(anchor="w")
        self.lbl_temp_caption = ttk.Label(inner, style="Tiny.TLabel")
        self.lbl_temp_caption.pack(anchor="w", pady=(4, 0))
        self._label(inner, "temp_note", style="Tiny.TLabel", pady=(6, 8))

    def _page_shift(self, parent):
        inner = self._page(parent)
        self._label(inner, "shift_intro", pady=(8, 4))
        caption = ttk.Label(inner, style="Head.TLabel")
        caption.pack(anchor="w")
        self.register(caption, "shift_lbl_c")
        self.shift_field = ValueField(inner, self, "shift_lbl_c", "", COL_RING,
                                      SHIFT_LO, SHIFT_HI, 1, DEFAULT_SHIFT,
                                      on_change=self._on_shift)
        self.shift_field.pack(fill="x", pady=(2, 6))

        grid = ttk.Frame(inner, style="Card.TFrame")
        grid.pack(fill="x", pady=(2, 4))
        grid.grid_columnconfigure(0, minsize=86)
        for col in range(1, N_CLASSES + 1):
            grid.grid_columnconfigure(col, minsize=94)
        self.shift_cols = []
        for k in range(N_CLASSES):
            head = ttk.Label(grid, style="Head.TLabel", anchor="e")
            head.configure(text=CLASS_TOKENS[k], foreground=CLASS_COLORS[k])
            head.grid(row=0, column=k + 1, sticky="e", padx=(0, 6), pady=(0, 3))
            self.shift_cols.append(head)
        rows = ("shift_head_orig", "shift_head_shifted", "shift_head_p",
                "shift_head_ps", "shift_head_diff")
        self.shift_rows = []
        for r, key in enumerate(rows, start=1):
            name = ttk.Label(grid, style="Tiny.TLabel", wraplength=84,
                             justify="left")
            name.grid(row=r, column=0, sticky="w", pady=1)
            self.register(name, key)
            cells = []
            for col in range(N_CLASSES):
                lbl = ttk.Label(grid, style="Mono.TLabel", width=9, anchor="e")
                lbl.grid(row=r, column=col + 1, sticky="e", padx=(0, 6), pady=1)
                cells.append(lbl)
            self.shift_rows.append(cells)

        self.lbl_shift_verdict = ttk.Label(inner, style="Tiny.TLabel",
                                           wraplength=PAGE_WRAP, justify="left")
        self.lbl_shift_verdict.pack(anchor="w", pady=(4, 0))
        self._label(inner, "shift_expl", style="Tiny.TLabel", pady=(6, 8))

    def _page_delta(self, parent):
        inner = self._page(parent)
        self._label(inner, "delta_intro", pady=(8, 4))
        caption = ttk.Label(inner, style="Head.TLabel")
        caption.pack(anchor="w")
        self.register(caption, "delta_lbl_class")
        self.var_delta_class = tk.IntVar(value=0)
        row = ttk.Frame(inner, style="Card.TFrame")
        row.pack(fill="x", pady=(2, 4))
        self.delta_radios = []
        for k in range(N_CLASSES):
            rb = ttk.Radiobutton(row, text="", variable=self.var_delta_class,
                                 value=k, command=self._on_delta_class,
                                 style="Radio%d.TRadiobutton" % k)
            rb.pack(side="left", padx=(0, 12))
            self.register(rb, CLASS_KEYS[k])
            self.delta_radios.append(rb)

        self.delta_field = ValueField(inner, self, "delta_lbl_d", "", COL_RING,
                                      DELTA_LO, DELTA_HI, 2, DEFAULT_DELTA,
                                      on_change=self._on_delta)
        self.delta_field.pack(fill="x", pady=(2, 6))

        grid = ttk.Frame(inner, style="Card.TFrame")
        grid.pack(fill="x", pady=(2, 4))
        heads = ("col_class", "delta_head_before", "delta_head_after",
                 "delta_head_diff")
        for col, key in enumerate(heads):
            lbl = ttk.Label(grid, style="Head.TLabel")
            lbl.grid(row=0, column=col, sticky="w" if col == 0 else "e",
                     padx=(0, 6))
            self.register(lbl, key)
        self.delta_rows = []
        for k in range(N_CLASSES):
            cells = []
            for col in range(len(heads)):
                lbl = ttk.Label(grid, style="Mono.TLabel",
                                width=5 if col == 0 else 10,
                                anchor="w" if col == 0 else "e")
                lbl.grid(row=k + 1, column=col, sticky="w" if col == 0 else "e",
                         padx=(0, 6), pady=1)
                cells.append(lbl)
            self.delta_rows.append(cells)

        self.delta_curves = DeltaCurves(inner, self, height=206)
        self.delta_curves.pack(fill="x", pady=(4, 2))
        self._label(inner, "delta_expl_coupled", style="Tiny.TLabel",
                    pady=(4, 8))

    def _page_ce(self, parent):
        inner = self._page(parent)
        self._label(inner, "ce_intro", pady=(8, 4))
        formula = ttk.Label(inner, style="MonoBold.TLabel", text=F_CE,
                            wraplength=PAGE_WRAP, justify="left")
        formula.pack(anchor="w", pady=(0, 6))

        caption = ttk.Label(inner, style="Head.TLabel")
        caption.pack(anchor="w")
        self.register(caption, "ce_lbl_target")
        self.var_ce_target = tk.IntVar(value=0)
        row = ttk.Frame(inner, style="Card.TFrame")
        row.pack(fill="x", pady=(2, 4))
        self.ce_radios = []
        for k in range(N_CLASSES):
            rb = ttk.Radiobutton(row, text="", variable=self.var_ce_target,
                                 value=k, command=self._on_ce_target,
                                 style="Radio%d.TRadiobutton" % k)
            rb.pack(side="left", padx=(0, 12))
            self.register(rb, CLASS_KEYS[k])
            self.ce_radios.append(rb)

        grid = ttk.Frame(inner, style="Card.TFrame")
        grid.pack(fill="x", pady=(2, 4))
        grid.grid_columnconfigure(0, minsize=86)
        for col in range(1, N_CLASSES + 1):
            grid.grid_columnconfigure(col, minsize=94)
        self.ce_cols = []
        for k in range(N_CLASSES):
            head = ttk.Label(grid, style="Head.TLabel", anchor="e",
                             text=CLASS_SHORT[k])
            head.grid(row=0, column=k + 1, sticky="e", padx=(0, 6), pady=(0, 3))
            self.ce_cols.append(head)
        self.ce_rows = []
        for r, key in enumerate(("ce_head_y", "ce_head_p", "ce_head_py"),
                                start=1):
            name = ttk.Label(grid, style="Tiny.TLabel", wraplength=84,
                             justify="left")
            name.grid(row=r, column=0, sticky="w", pady=1)
            self.register(name, key)
            cells = []
            for col in range(N_CLASSES):
                lbl = ttk.Label(grid, style="Mono.TLabel", width=9, anchor="e")
                lbl.grid(row=r, column=col + 1, sticky="e", padx=(0, 6), pady=1)
                cells.append(lbl)
            self.ce_rows.append(cells)

        calc = ttk.Frame(inner, style="Card.TFrame")
        calc.pack(fill="x", pady=(2, 4))
        calc.grid_columnconfigure(0, minsize=150)
        calc.grid_columnconfigure(1, minsize=96)
        self.ce_calc = []
        for r, key in enumerate(("ce_row_lnS", "ce_row_u", "ce_row_ce")):
            style = "Head.TLabel" if r == 2 else "Tiny.TLabel"
            name = ttk.Label(calc, style=style, wraplength=146, justify="left")
            name.grid(row=r, column=0, sticky="w", pady=1)
            self.register(name, key)
            lbl = ttk.Label(calc, style="MonoBold.TLabel" if r == 2
                            else "Mono.TLabel", anchor="e")
            lbl.grid(row=r, column=1, sticky="e", pady=1)
            self.ce_calc.append(lbl)

        self.lbl_ce_verdict = ttk.Label(inner, style="Key.TLabel",
                                        wraplength=PAGE_WRAP, justify="left")
        self.lbl_ce_verdict.pack(anchor="w", pady=(4, 0))
        self.lbl_ce_sum = ttk.Label(inner, style="Tiny.TLabel",
                                    wraplength=PAGE_WRAP, justify="left")
        self.lbl_ce_sum.pack(anchor="w", pady=(4, 0))
        self._label(inner, "ce_expl_grad", style="Tiny.TLabel", pady=(6, 2))
        self._label(inner, "ce_expl_zero", style="Tiny.TLabel", pady=(0, 8))

    def _page_stable(self, parent):
        inner = self._page(parent)
        self._label(inner, "stable_intro", pady=(8, 4))
        caption = ttk.Label(inner, style="Head.TLabel")
        caption.pack(anchor="w")
        self.register(caption, "stable_lbl_case")
        self.var_case = tk.StringVar()
        self.case_combo = ttk.Combobox(inner, textvariable=self.var_case,
                                       state="readonly", width=42,
                                       style="Case.TCombobox")
        self.case_combo.pack(anchor="w", pady=(2, 6))
        self.case_combo.bind("<<ComboboxSelected>>", self._on_case)

        grid = ttk.Frame(inner, style="Card.TFrame")
        grid.pack(fill="x", pady=(2, 4))
        grid.grid_columnconfigure(0, minsize=104)
        for col in range(1, N_CLASSES + 1):
            grid.grid_columnconfigure(col, minsize=94)
        for k in range(N_CLASSES):
            head = ttk.Label(grid, style="Head.TLabel", anchor="e")
            head.configure(text=CLASS_TOKENS[k], foreground=CLASS_COLORS[k])
            head.grid(row=0, column=k + 1, sticky="e", padx=(0, 6), pady=(0, 3))
        rows = (("stable_head_z", "class"), ("stable_head_naive", "class"),
                ("stable_head_safe", "class"), ("stable_row_sum_naive", "scalar"),
                ("stable_row_sum_safe", "scalar"), ("stable_row_p_naive", "class"),
                ("stable_row_p_safe", "class"),
                ("stable_row_verdict_naive", "scalar"),
                ("stable_row_verdict_safe", "scalar"))
        self.stable_rows = []
        for r, (key, shape) in enumerate(rows, start=1):
            name = ttk.Label(grid, style="Tiny.TLabel", wraplength=100,
                             justify="left")
            name.grid(row=r, column=0, sticky="w", pady=1)
            self.register(name, key)
            cells = []
            if shape == "class":
                for col in range(N_CLASSES):
                    lbl = ttk.Label(grid, style="Mono.TLabel", width=9,
                                    anchor="e")
                    lbl.grid(row=r, column=col + 1, sticky="e", padx=(0, 6),
                             pady=1)
                    cells.append(lbl)
            else:
                lbl = ttk.Label(grid, style="Mono.TLabel", anchor="e")
                lbl.grid(row=r, column=1, columnspan=N_CLASSES, sticky="e",
                         padx=(0, 6), pady=1)
                cells.append(lbl)
            self.stable_rows.append(cells)

        self.lbl_stable_naive = ttk.Label(inner, style="Key.TLabel",
                                          wraplength=PAGE_WRAP, justify="left")
        self.lbl_stable_naive.pack(anchor="w", pady=(4, 0))
        self.lbl_stable_safe = ttk.Label(inner, style="Key.TLabel",
                                         wraplength=PAGE_WRAP, justify="left")
        self.lbl_stable_safe.pack(anchor="w", pady=(2, 0))
        self.lbl_stable_limit = ttk.Label(inner, style="Tiny.TLabel",
                                          wraplength=PAGE_WRAP, justify="left")
        self.lbl_stable_limit.pack(anchor="w", pady=(6, 8))

    def _page_steps(self, parent):
        inner = self._page(parent)
        self._label(inner, "steps_intro", style="Tiny.TLabel", pady=(8, 4))
        self.chip_labels = []
        chips_a = ttk.Frame(inner, style="Card.TFrame")
        chips_a.pack(fill="x", pady=(0, 1))
        chips_b = ttk.Frame(inner, style="Card.TFrame")
        chips_b.pack(fill="x", pady=(0, 6))
        for index, key in enumerate(STEP_CHIPS):
            holder = chips_a if index < 4 else chips_b
            if index:
                sep = ttk.Label(holder, style="Tiny.TLabel")
                sep.configure(text="->")
                sep.pack(side="left", padx=2)
            chip = ttk.Label(holder, style="Chip.TLabel", padding=(5, 2))
            chip.pack(side="left")
            self.register(chip, key)
            self.chip_labels.append(chip)

        self.lbl_step_of = ttk.Label(inner, style="Head.TLabel")
        self.lbl_step_of.pack(anchor="w", pady=(2, 0))
        self.lbl_step_title = ttk.Label(inner, style="Head.TLabel",
                                        foreground=COL_KEY)
        self.lbl_step_title.pack(anchor="w", pady=(4, 0))
        self.lbl_step_formula = ttk.Label(inner, style="MonoBold.TLabel",
                                          wraplength=PAGE_WRAP, justify="left")
        self.lbl_step_formula.pack(anchor="w", pady=(2, 0))
        self.lbl_step_explain = ttk.Label(inner, style="Key.TLabel",
                                          wraplength=PAGE_WRAP, justify="left")
        self.lbl_step_explain.pack(anchor="w", pady=(4, 6))

        buttons = ttk.Frame(inner, style="Card.TFrame")
        buttons.pack(fill="x", pady=(2, 8))
        prev = ttk.Button(buttons, width=10, command=self.step_prev)
        prev.pack(side="left")
        self.register(prev, "btn_prev")
        nxt = ttk.Button(buttons, width=10, command=self.step_next)
        nxt.pack(side="left", padx=(6, 0))
        self.register(nxt, "btn_next")
        rst = ttk.Button(buttons, width=10, command=self.step_reset)
        rst.pack(side="left", padx=(6, 0))
        self.register(rst, "btn_reset_steps")
        self.step_buttons = (prev, nxt, rst)

    def _page_help(self, parent):
        inner = self._page(parent)
        self._label(inner, "help_title", style="CardHead.TLabel", pady=(8, 4))
        self._label(inner, "tie_rule", style="Key.TLabel", pady=(2, 4))
        self._label(inner, "calib_note", style="Tiny.TLabel", pady=(2, 6))
        for index in range(len(HELP_ITEMS)):
            title_key = "h%d_t" % (index + 1)
            body_key = "h%d_b" % (index + 1)
            self._label(inner, title_key, style="Head.TLabel", pady=(7, 1))
            self._label(inner, body_key, style="Tiny.TLabel", pady=(0, 0))

    # --- нижняя строка ------------------------------------------------------

    def _build_statusbar(self):
        bar = ttk.Frame(self, style="Body.TFrame")
        bar.pack(fill="x", padx=10, pady=(0, 6))
        self.lbl_status = ttk.Label(bar, style="Status.TLabel")
        self.lbl_status.pack(side="left", fill="x", expand=False)
        author = ttk.Label(bar, text=AUTHOR, style="Author.TLabel")
        author.pack(side="right", padx=(8, 0))
        author.lift()

    # --- реакция на ввод ----------------------------------------------------

    def _on_logit(self, index, value):
        self.logits[index] = value
        self.refresh()

    @staticmethod
    def _temp_is_positive(value):
        """Температура — делитель: нужно число строго больше нуля."""
        return None if value > 0 else "status_bad_temp"

    def _on_temp(self, value):
        self.temperature = value
        self.refresh()
        self.set_status("status_temp", t=fmt(value, 2),
                        hint=T(self._temp_hint(value)))

    @staticmethod
    def _temp_hint(value):
        if value < 1.0:
            return "temp_sharp"
        if value == 1.0:
            return "temp_standard"
        return "temp_flat"

    def _on_shift(self, value):
        self.shift = value
        self.refresh()
        self.set_status("status_shift", c=fmt(value, 1))

    def _on_delta(self, value):
        self.delta = value
        self.refresh()
        self._announce_delta()

    def _on_delta_class(self):
        if getattr(self, "_building", True):
            return
        self.delta_class = int(self.var_delta_class.get())
        self.refresh()
        self._announce_delta()

    def _announce_delta(self):
        self.set_status("status_delta", name=T(CLASS_KEYS[self.delta_class]),
                        d=fmt(self.delta, 2))

    def _on_ce_target(self):
        if getattr(self, "_building", True):
            return
        self.ce_target = int(self.var_ce_target.get())
        self.refresh()
        self.set_status("status_ce", name=T(CLASS_KEYS[self.ce_target]))

    def _on_case(self, _event=None):
        if getattr(self, "_building", True):
            return
        text = self.var_case.get()
        for index, (key, _values) in enumerate(STABILITY_CASES):
            if T(key) == text:
                self.stab_case = index
                break
        self.refresh()
        self.set_status("status_case", name=text)

    def _on_tab(self, _event=None):
        if getattr(self, "_building", True):
            return
        try:
            index = self.nb.index(self.nb.select())
        except (tk.TclError, IndexError):
            return
        if index >= len(self._tab_tags):
            return
        self.active_tab = self._tab_tags[index]
        self.refresh()

    def apply_preset(self, index):
        key, values = PRESETS[index]
        self.logits = list(values)
        for k, field in enumerate(self.logit_fields):
            field.set_value(self.logits[k])
        self.refresh()
        self.set_status("status_preset", name=T(key))

    def reset_all(self):
        self.logits = list(DEFAULT_LOGITS)
        self.temperature = DEFAULT_TEMP
        self.shift = DEFAULT_SHIFT
        self.delta = DEFAULT_DELTA
        self.delta_class = 0
        self.ce_target = 0
        self.stab_case = 0
        self.step = STEP_COUNT
        self.trail = []
        for k, field in enumerate(self.logit_fields):
            field.set_value(self.logits[k])
        self.temp_field.set_value(self.temperature)
        self.shift_field.set_value(self.shift)
        self.delta_field.set_value(self.delta)
        if hasattr(self, "var_delta_class"):
            self.var_delta_class.set(0)
        if hasattr(self, "var_ce_target"):
            self.var_ce_target.set(self.ce_target)
        self.refresh()
        self.set_status("status_reset")

    def step_move(self, offset):
        self.step = clamp(int(self.step + offset), 1, STEP_COUNT)
        self.refresh()
        self.set_status("status_step", n=self.step, total=STEP_COUNT,
                        title=T(STEP_TITLES[self.step - 1]))

    def step_next(self):
        self.step_move(1)

    def step_prev(self):
        self.step_move(-1)

    def step_reset(self):
        self.step = STEP_COUNT
        self.refresh()
        self.set_status("status_step", n=self.step, total=STEP_COUNT,
                        title=T(STEP_TITLES[self.step - 1]))

    def _on_arrow(self, event):
        try:
            cls = event.widget.winfo_class()
        except tk.TclError:
            return
        if cls in ("TEntry", "Entry", "TCombobox", "Text", "TNotebook",
                   "TNotebook.Tab"):
            return
        offset = 1 if event.keysym == "Right" else -1
        self.step_move(offset)

    # --- пересчёт и перерисовка ---------------------------------------------

    def refresh(self):
        self.chain = softmax_chain(self.logits, self.temperature)
        new = self.chain["probs"]
        if new != self.probs:
            self.trail.append(list(self.probs))
            if len(self.trail) > TRAIL_MAX:
                del self.trail[:-TRAIL_MAX]
            self.probs = list(new)
        self.best = argmax_index(self.probs)
        self._update_ring()
        self.simplex.redraw()
        self.stack.redraw()
        self.scale_view.redraw()
        self.delta_curves.redraw()
        self._paint_dist()
        self._paint_chain()
        self._paint_temp()
        self._paint_shift()
        self._paint_delta()
        self._paint_ce()
        self._paint_stable()
        self._paint_steps()

    def _update_ring(self):
        if self.active_tab == "shift" and self.shift != 0.0:
            self.ring = softmax(add_constant(self.logits, self.shift),
                                self.temperature)
            self.ring_key = "lbl_ring_shift"
        elif self.active_tab == "delta" and self.delta != 0.0:
            self.ring = softmax(perturb_one(self.logits, self.delta_class,
                                            self.delta), self.temperature)
            self.ring_key = "lbl_ring_delta"
        else:
            self.ring = None
            self.ring_key = None

    def _paint_dist(self):
        probs = self.probs
        best = self.best
        self.dist_values[0].configure(text=CLASS_SHORT[best],
                                      foreground=CLASS_COLORS[best])
        self.dist_values[1].configure(text=fmt_prob(probs[best]))
        self.dist_values[2].configure(text=fmt_precise(math.fsum(probs)))
        value = entropy(probs)
        self.lbl_ent_value.configure(text=fmt(value, 4))
        self.ent_bar.redraw(value)
        if is_tie(probs, best):
            if not self.lbl_tie.winfo_manager():
                self.lbl_tie.pack(anchor="w", pady=(4, 0))
        elif self.lbl_tie.winfo_manager():
            self.lbl_tie.pack_forget()
        if self.ring is not None and self.ring_key:
            self.lbl_ring.configure(text=T(self.ring_key))
            if not self.lbl_ring.winfo_manager():
                self.lbl_ring.pack(anchor="w")
        elif self.lbl_ring.winfo_manager():
            self.lbl_ring.pack_forget()
        if self.trail:
            if not self.lbl_trail.winfo_manager():
                self.lbl_trail.pack(anchor="w")
        elif self.lbl_trail.winfo_manager():
            self.lbl_trail.pack_forget()

    def _chain_data(self, kind):
        if kind == "logits":
            return self.logits
        if kind == "scaled":
            return self.chain["scaled"]
        if kind == "shifted":
            return self.chain["shifted"]
        if kind == "exps":
            return self.chain["exps"]
        if kind == "probs":
            return self.chain["probs"]
        if kind == "m":
            return [self.chain["max"]]
        if kind == "denom":
            return [self.chain["denom"]]
        return [self.best]

    def _chain_text(self, kind, value):
        if kind == "pred":
            return CLASS_SHORT[value] + ("  *" if is_tie(self.probs, value)
                                         else "")
        if kind in ("exps", "denom", "probs"):
            return fmt_tiny(value)
        return fmt_chain(value)

    def _paint_chain(self):
        for row in self.chain_rows:
            stage = row["stage"]
            revealed = stage <= self.step
            if stage == self.step:
                style = "Hi.TLabel"
            elif revealed:
                style = "Mono.TLabel"
            else:
                style = "Dim.TLabel"
            row["name"].configure(style=style)
            data = self._chain_data(row["kind"]) if revealed else None
            for index, lbl in enumerate(row["labels"]):
                lbl.configure(style=style,
                              text="—" if data is None
                              else self._chain_text(row["kind"], data[index]))

    def _paint_temp(self):
        self.temp_compare.refresh()
        self.lbl_temp_caption.configure(text="*  " + T("temp_tag_current"))

    def _paint_shift(self):
        shifted = add_constant(self.logits, self.shift)
        after = softmax(shifted, self.temperature)
        before = self.probs
        for k in range(N_CLASSES):
            self.shift_rows[0][k].configure(text=fmt_chain(self.logits[k]))
            self.shift_rows[1][k].configure(text=fmt_chain(shifted[k]))
            self.shift_rows[2][k].configure(text=fmt_tiny(before[k]))
            self.shift_rows[3][k].configure(text=fmt_tiny(after[k]))
            self.shift_rows[4][k].configure(
                text=fmt_gap(abs(before[k] - after[k])))
        diff = max_abs_diff(before, after)
        if diff <= SHIFT_TOL:
            self.lbl_shift_verdict.configure(
                text=T("shift_verdict_same", diff=fmt_precise(diff)),
                foreground=COL_GOOD)
        else:
            self.lbl_shift_verdict.configure(
                text=T("shift_verdict_diff", diff=fmt_precise(diff)),
                foreground=COL_BAD_FG)

    def _paint_delta(self):
        after = softmax(perturb_one(self.logits, self.delta_class, self.delta),
                        self.temperature)
        for k in range(N_CLASSES):
            cells = self.delta_rows[k]
            active = k == self.delta_class
            cells[0].configure(text=CLASS_TOKENS[k],
                               foreground=CLASS_COLORS[k] if active else COL_TEXT)
            cells[1].configure(text=fmt_tiny(self.probs[k]))
            cells[2].configure(text=fmt_tiny(after[k]))
            cells[3].configure(text="%+.6f" % (after[k] - self.probs[k]))

    def _paint_ce(self):
        target = self.ce_target
        y = one_hot(target)
        diff = probs_minus_target(self.chain, target)
        loss = cross_entropy(self.chain, target)
        for k in range(N_CLASSES):
            self.ce_cols[k].configure(foreground=CLASS_COLORS[k]
                                      if k == target else COL_DIM)
            self.ce_rows[0][k].configure(text="%d" % y[k],
                                        foreground=COL_KEY if k == target
                                        else COL_TEXT)
            self.ce_rows[1][k].configure(text=fmt_tiny(self.probs[k]))
            self.ce_rows[2][k].configure(text=fmt_signed(diff[k]))
        self.ce_calc[0].configure(text=fmt(math.log(self.chain["denom"]), 6))
        self.ce_calc[1].configure(text=fmt_chain(self.chain["shifted"][target]))
        self.ce_calc[2].configure(text=fmt(loss, 6))
        if self.best == target:
            self.lbl_ce_verdict.configure(
                text=T("ce_verdict_match", name=T(CLASS_KEYS[target])),
                foreground=COL_GOOD)
        else:
            self.lbl_ce_verdict.configure(
                text=T("ce_verdict_miss", name=T(CLASS_KEYS[target]),
                       best=CLASS_SHORT[self.best]), foreground=COL_BAD_FG)
        total = math.fsum(diff)
        self.lbl_ce_sum.configure(
            text=T("ce_sum_line", v=fmt_precise(total)),
            foreground=COL_GOOD if abs(total) <= SHIFT_TOL else COL_BAD_FG)

    def _paint_stable(self):
        key, values = STABILITY_CASES[self.stab_case]
        logits = list(self.logits) if values is None else list(values)
        naive = naive_chain(logits, self.temperature)
        safe = softmax_chain(logits, self.temperature)
        self.case_combo.configure(values=[T(k) for k, _v in STABILITY_CASES])
        self.var_case.set(T(key))

        rows = self.stable_rows
        naive_res = "st_" + naive["status"]
        naive_ok = naive["status"] == "finite"
        for k in range(N_CLASSES):
            flag = naive["flags"][k]
            rows[0][k].configure(text=fmt(logits[k], 1))
            rows[1][k].configure(text=fmt_tiny(naive["exps"][k]),
                                 foreground=COL_TEXT if flag == "finite"
                                 else COL_BAD_FG)
            rows[2][k].configure(text=fmt_tiny(safe["exps"][k]),
                                 foreground=COL_GOOD)
            if naive["probs"] is None:
                rows[5][k].configure(text="—", foreground=COL_BAD_FG)
            else:
                rows[5][k].configure(text=fmt_tiny(naive["probs"][k]),
                                     foreground=COL_TEXT)
            rows[6][k].configure(text=fmt_tiny(safe["probs"][k]),
                                 foreground=COL_GOOD)
        rows[3][0].configure(text=fmt(naive["denom"]),
                             foreground=COL_TEXT if naive_ok else COL_BAD_FG)
        rows[4][0].configure(text=fmt(safe["denom"], 4), foreground=COL_GOOD)
        rows[7][0].configure(text=T(naive_res),
                             foreground=COL_TEXT if naive_ok else COL_BAD_FG)
        rows[8][0].configure(text=T("st_finite"), foreground=COL_GOOD)
        self.lbl_stable_naive.configure(text=T("stable_verdict_naive",
                                               res=T(naive_res)))
        self.lbl_stable_safe.configure(text=T("stable_verdict_safe",
                                              res=T("st_finite")))
        self.lbl_stable_limit.configure(text=T("stable_note_limit",
                                               v=fmt(EXP_LIMIT, 2)))

    def _paint_steps(self):
        step = self.step
        self.lbl_step_of.configure(text=T("step_of", n=step, total=STEP_COUNT))
        self.lbl_step_title.configure(text=T(STEP_TITLES[step - 1]))
        self.lbl_step_formula.configure(text=STEP_FORMULAS[step - 1])
        self.lbl_step_explain.configure(text=T(STEP_EXPLAINS[step - 1]))
        for index, chip in enumerate(self.chip_labels):
            stage = index + 1
            if stage == step:
                style = "ChipHi.TLabel"
            elif stage < step:
                style = "Chip.TLabel"
            else:
                style = "ChipDim.TLabel"
            chip.configure(style=style)
        prev, nxt, rst = self.step_buttons
        # Начальное состояние шагов — последний шаг, поэтому «Сбросить»
        # неактивен именно там: сбрасывать с него нечего.
        prev.state(["disabled" if step == 1 else "!disabled"])
        nxt.state(["disabled" if step == STEP_COUNT else "!disabled"])
        rst.state(["disabled" if step == STEP_COUNT else "!disabled"])


# ---------------------------------------------------------------------------
# Раздел 12. Точка входа
# ---------------------------------------------------------------------------


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()


