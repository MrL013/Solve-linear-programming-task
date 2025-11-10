from __future__ import annotations
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

EPS = 1e-9  # числовой допуск для сравнения с нулем

@dataclass
class LPParsed:
    """Структура для хранения исходных данных задачи."""
    sense: str                 # тип задачи: "min" или "max"
    c: List[float]             # коэффициенты целевой функции для x1..xn
    constraints: List[Tuple[List[float], str, float]]  # (коэффициенты, знак, правая часть)
    num_vars: int              # число исходных переменных
    free_vars: set[int]        # индексы (1-based) свободных переменных


def _strip_spaces(expr: str) -> str:
    """Убирает лишние пробелы и заменяет минусы на стандартный ASCII-символ."""
    expr = expr.replace("−", "-").replace("–", "-")
    expr = re.sub(r"[ \t]+", " ", expr)
    return expr.strip()


def _parse_linear_comb(s: str, num_hint: int | None = None) -> Tuple[List[float], int]:
    """
    Разбирает строку линейной комбинации, например:
        "3x1 - 2 x2 + x4"
    Возвращает список коэффициентов и максимальный индекс переменной.
    """
    s = s.replace("−", "-")
    # добавляем коэффициент 1, если он не указан явно
    s = re.sub(r"(^|\s|\+|\-)(?=\s*x\d+)", r"\g<1> 1*", s)
    s = s.replace("*", "")
    terms = re.finditer(r"([+-]?\s*\d+(?:\.\d+)?)(?:\s*)x(\d+)", s, flags=re.I)
    coeffs: Dict[int, float] = {}
    maxj = 0
    for m in terms:
        a = float(m.group(1).replace(" ", ""))
        j = int(m.group(2))
        coeffs[j] = coeffs.get(j, 0.0) + a
        maxj = max(maxj, j)
    n = max(num_hint or 0, maxj)
    vec = [0.0] * n
    for j, a in coeffs.items():
        vec[j - 1] = a
    return vec, n


def parse_problem(text: str) -> LPParsed:
    """
    Парсит текстовую постановку задачи.
    Поддерживает секции:
      Objective: max: ...
      Subject To:
         ...
      Bounds: x2 free, x4 free
    """
    text = _strip_spaces(text)

    #  Целевая функция
    m = re.search(r"(?mi)^objective\s*:?\s*(min|max)\s*:\s*(.+)$", text)
    if not m:
        m = re.search(r"(?mi)^(min|max)\s*:\s*(.+)$", text)
    if not m:
        raise ValueError("Не найдена строка цели. Ожидается, например: 'Objective: max: 3x1 + 2x2'.")

    sense = m.group(1).lower()
    c_vec, n = _parse_linear_comb(m.group(2))

    # Ограничения
    block = re.search(r"(?mis)^subject\s*to\s*:?(.*?)(?:^\w|$\Z)", text)
    cons_text = ""
    if block:
        cons_text = block.group(1)
    else:
        # fallback: берём все строки с <=, >= или =
        lines = [ln for ln in text.splitlines() if re.search(r"(<=|≥|>=|=|≤)", ln)]
        cons_text = "\n".join(lines)

    constraints: List[Tuple[List[float], str, float]] = []
    for line in cons_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.replace("≤", "<=").replace("≥", ">=")
        mo = re.search(r"(.*?)(<=|>=|=)(.*)$", line)
        if not mo:
            continue
        lhs, rel, rhs = mo.group(1).strip(), mo.group(2), float(mo.group(3))
        coeffs, n2 = _parse_linear_comb(lhs, n)
        n = max(n, n2)
        constraints.append((coeffs, rel, rhs))

    # Секция Bounds (свободные переменные)
    free_vars: set[int] = set()
    bmatch = re.search(r"(?mis)^bounds\s*:(.*?)(?:^\w|$\Z)", text)
    if bmatch:
        chunk = bmatch.group(1)
        for token in re.split(r",", chunk):
            token = token.strip()
            mm = re.match(r"x(\d+)\s+free", token, flags=re.I)
            if mm:
                j = int(mm.group(1))
                free_vars.add(j)

    # дополняем коэффициенты до полной длины
    c_vec = (c_vec + [0.0] * n)[:n]
    return LPParsed(sense=sense, c=c_vec, constraints=constraints, num_vars=n, free_vars=free_vars)


@dataclass
class ExpansionMap:
    """Отображение между расширенными и исходными переменными."""
    cols: List[Tuple[int, bool]]  # (индекс исходной, True если x⁺, False если x⁻)
    n_orig: int                   # число исходных переменных

    def collapse(self, x_ext: List[float]) -> List[float]:
        """Сворачивает решение из пространства (x⁺, x⁻) обратно в исходные x."""
        x = [0.0] * self.n_orig
        for k, (j, is_plus) in enumerate(self.cols):
            if is_plus:
                x[j] += x_ext[k]
            else:
                x[j] -= x_ext[k]
        return [0.0 if abs(v) < 1e-9 else v for v in x]


def expand_unrestricted(c: List[float],
                        constraints: List[Tuple[List[float], str, float]],
                        free: set[int]) -> Tuple[List[float], List[Tuple[List[float], str, float]], ExpansionMap]:
    """
    Для каждой свободной переменной x_j создает две неотрицательные:
        x_j = x_j⁺ - x_j⁻,   x_j⁺ ≥ 0, x_j⁻ ≥ 0
    Расширяет матрицу ограничений и вектор цели.
    """
    n = len(c)
    new_c: List[float] = []
    mapping_cols: List[Tuple[int, bool]] = []

    #  Целевая функция
    for j in range(n):
        if (j + 1) in free:
            new_c.extend([c[j], -c[j]])  # xj⁺, xj⁻
            mapping_cols.append((j, True))
            mapping_cols.append((j, False))
        else:
            new_c.append(c[j])
            mapping_cols.append((j, True))

    #  Ограничения
    new_cons: List[Tuple[List[float], str, float]] = []
    for coeffs, rel, rhs in constraints:
        row: List[float] = []
        for j in range(n):
            a = coeffs[j] if j < len(coeffs) else 0.0
            if (j + 1) in free:
                row.extend([a, -a])
            else:
                row.append(a)
        new_cons.append((row, rel, rhs))

    return new_c, new_cons, ExpansionMap(cols=mapping_cols, n_orig=n)


@dataclass
class Tableau:
    """
    Симплекс-таблица для системы Ax = b, x ≥ 0.
    """
    a: List[List[float]]
    b: List[float]
    c: List[float]
    v: float
    basis: List[int]
    var_names: List[str]

    def shape(self) -> Tuple[int, int]:
        return len(self.b), len(self.c)

    def pivot(self, row: int, col: int):
        """Выполняет элементарное преобразование таблицы (поворот по элементу (row, col))."""
        m, n = self.shape()
        piv = self.a[row][col]
        if abs(piv) < EPS:
            raise RuntimeError("Нулевой разрешающий элемент.")
        inv = 1.0 / piv
        self.a[row] = [v * inv for v in self.a[row]]
        self.b[row] *= inv
        for i in range(m):
            if i == row:
                continue
            factor = self.a[i][col]
            if abs(factor) < EPS:
                continue
            self.a[i] = [self.a[i][j] - factor * self.a[row][j] for j in range(n)]
            self.b[i] -= factor * self.b[row]
        factor = self.c[col]
        if abs(factor) > EPS:
            self.c = [self.c[j] - factor * self.a[row][j] for j in range(n)]
            self.v -= factor * self.b[row]
        self.basis[row] = col

    def arg_enter_bland(self) -> Optional[int]:
        """Выбор входящей переменной по правилу Бланда (первая с c_j > 0)."""
        for j, cj in enumerate(self.c):
            if cj > EPS:
                return j
        return None

    def arg_leave_min_ratio(self, col: int) -> Optional[int]:
        """Правило минимального отношения (min b_i / a_ij)."""
        candidates = []
        for i, aic in enumerate(self.a):
            if aic[col] > EPS:
                candidates.append((self.b[i] / aic[col], self.basis[i], i))
        if not candidates:
            return None
        _, _, row = min(candidates, key=lambda t: (t[0], t[1]))
        return row


@dataclass
class StandardForm:
    """Каноническая форма задачи: Ax=b, x>=0."""
    A: List[List[float]]
    b: List[float]
    c: List[float]
    var_names: List[str]
    slack_idx: List[int]
    artificial_idx: List[int]


def to_standard_form(cons: List[Tuple[List[float], str, float]],
                     c: List[float]) -> StandardForm:
    """
    Приводит задачу к канонической форме, добавляя
    недостающие переменные (slack, surplus, artificial).
    При b_i < 0 строки нормализуются (умножаются на −1).
    """
    m = len(cons)
    n = len(c)
    A: List[List[float]] = []
    b: List[float] = []
    var_names = [f"x{j+1}" for j in range(n)]
    slack_idx: List[int] = []
    artificial_idx: List[int] = []

    for coeffs, rel, rhs in cons:
        row = (coeffs + [0.0] * n)[:n]
        A.append(row)
        b.append(rhs)

    # Нормализация строк с отрицательными правыми частями
    for i in range(m):
        if b[i] < -EPS:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]
            cons[i] = (cons[i][0], {"<=": ">=", ">=": "<=", "=": "="}[cons[i][1]], -cons[i][2])

    # Добавление дополнительных переменных
    for i, (_, rel, _) in enumerate(cons):
        if rel == "<=":
            for r in range(m):
                A[r].append(1.0 if r == i else 0.0)
            slack_idx.append(n); var_names.append(f"s{i+1}")
            n += 1
        elif rel == ">=":
            for r in range(m):
                A[r].append(-1.0 if r == i else 0.0)
            slack_idx.append(n); var_names.append(f"s{i+1}")
            n += 1
            for r in range(m):
                A[r].append(1.0 if r == i else 0.0)
            artificial_idx.append(n); var_names.append(f"a{i+1}")
            n += 1
        elif rel == "=":
            for r in range(m):
                A[r].append(1.0 if r == i else 0.0)
            artificial_idx.append(n); var_names.append(f"a{i+1}")
            n += 1

    return StandardForm(A=A, b=b, c=c, var_names=var_names,
                        slack_idx=slack_idx, artificial_idx=artificial_idx)


def build_phase1_tableau(sf: StandardForm) -> Tableau:
    """Создаёт таблицу для Фазы I (минимизация суммы искусственных переменных)."""
    m = len(sf.b)
    n = len(sf.var_names)
    a = [row[:] for row in sf.A]
    b = sf.b[:]
    c = [0.0] * n
    for idx in sf.artificial_idx:
        c[idx] = 1.0  # цель: min Σ a_i  (при решении max меняем знак)
    basis = [-1] * m
    for i in range(m):
        found = False
        for idx in sf.artificial_idx:
            if abs(a[i][idx] - 1.0) < EPS and all(abs(a[r][idx]) < EPS for r in range(m) if r != i):
                basis[i] = idx; found = True; break
        if found:
            continue
        for idx in sf.slack_idx:
            if abs(a[i][idx] - 1.0) < EPS and all(abs(a[r][idx]) < EPS for r in range(m) if r != i):
                basis[i] = idx; break
        if basis[i] == -1:
            a[i] += [0.0]
            for r in range(m):
                a[r].append(1.0 if r == i else 0.0)
            c.append(1.0)
            sf.var_names.append(f"a_extra{i+1}")
            idx_new = len(a[i]) - 1
            basis[i] = idx_new
    c = [-v for v in c]
    v = 0.0
    for i in range(m):
        bi = basis[i]
        if bi in sf.artificial_idx or sf.var_names[bi].startswith("a_extra"):
            c = [c[j] + a[i][j] for j in range(len(c))]
            v += b[i]
    return Tableau(a=a, b=b, c=c, v=-v, basis=basis, var_names=sf.var_names[:])


def simplex(tableau: Tableau, max_iter: int = 10_000) -> Tuple[str, Tableau, int]:
    """Реализация стандартного симплекс-алгоритма (максимизация)."""
    it = 0
    while it < max_iter:
        it += 1
        col = tableau.arg_enter_bland()
        if col is None:
            return "optimal", tableau, it
        row = tableau.arg_leave_min_ratio(col)
        if row is None:
            return "unbounded", tableau, it
        tableau.pivot(row, col)
    return "iterations_exceeded", tableau, it


def phase1(sf: StandardForm, verbose: bool = True) -> Tuple[str, Tableau, int]:
    """Фаза I — поиск допустимого базиса."""
    tab = build_phase1_tableau(sf)
    if verbose:
        print("=== ФАЗА I: поиск допустимого базиса ===")
    status, tab, it = simplex(tab)
    if status != "optimal":
        return status, tab, it
    if tab.v < -EPS:
        return "infeasible", tab, it
    # удаляем искусственные переменные
    keep_cols = [j for j, nm in enumerate(tab.var_names) if not nm.startswith("a")]
    a2 = [[row[j] for j in keep_cols] for row in tab.a]
    c2 = [0.0] * len(keep_cols)
    var_names2 = [tab.var_names[j] for j in keep_cols]
    mapping = {new_j: old_j for new_j, old_j in enumerate(keep_cols)}
    basis2 = [next((new_j for new_j, old_j in mapping.items() if old_j == bi), -1) for bi in tab.basis]
    tab2 = Tableau(a=a2, b=tab.b[:], c=c2, v=0.0, basis=basis2, var_names=var_names2)
    return "feasible", tab2, it


def phase2(tab: Tableau, c_original: List[float], verbose: bool = True) -> Tuple[str, Tableau, int]:
    """Фаза II — оптимизация исходной целевой функции."""
    m, n = tab.shape()
    tab.c = c_original[:] + [0.0] * (n - len(c_original))
    tab.v = 0.0
    # Обнуляем коэффициенты по базисным переменным в строке цели
    for i in range(m):
        bi = tab.basis[i]
        coeff = tab.c[bi]
        if abs(coeff) > EPS:
            tab.c = [tab.c[j] - coeff * tab.a[i][j] for j in range(n)]
            tab.v -= coeff * tab.b[i]
            tab.c[bi] = 0.0
    if verbose:
        print("=== ФАЗА II: оптимизация исходной цели ===")
    return simplex(tab)


def solve_from_text(text: str, verbose: bool = True) -> dict:
    """Главная функция решения ЗЛП из текстовой постановки."""
    t0 = time.time()
    parsed = parse_problem(text)

    # Разворачиваем свободные переменные
    c_ext, cons_ext, exp_map = expand_unrestricted(parsed.c, parsed.constraints, parsed.free_vars)

    # Подготовка цели (задачу min превращаем в max путём умножения на -1)
    if parsed.sense == "min":
        c_goal = [-v for v in c_ext]
        sense_mult = -1.0
    else:
        c_goal = c_ext[:]
        sense_mult = 1.0

    # Каноническая форма
    sf = to_standard_form(cons_ext, c_goal)

    # Фаза I
    st1, tab1, it1 = phase1(sf, verbose=verbose)
    if st1 in ("unbounded", "iterations_exceeded"):
        return {"status": st1, "iterations": it1, "time_sec": time.time() - t0}
    if st1 == "infeasible":
        return {"status": "infeasible", "iterations": it1, "time_sec": time.time() - t0}

    # Фаза II
    st2, tab2, it2 = phase2(tab1, c_goal, verbose=verbose)
    if st2 != "optimal":
        return {"status": st2, "iterations": it1 + it2, "time_sec": time.time() - t0}

    # Восстанавливаем вектор решения (в расширенных координатах)
    m, n = tab2.shape()
    x_ext = [0.0] * n
    for i in range(m):
        bi = tab2.basis[i]
        if 0 <= bi < n:
            x_ext[bi] = tab2.b[i]

    # Оставляем только "истинные" x-столбцы (убираем слэки)
    keep_cols = [j for j, nm in enumerate(tab2.var_names) if nm.startswith("x")]
    x_ext_main = [x_ext[j] for j in keep_cols]

    # Сворачиваем обратно к исходным переменным
    x_orig = exp_map.collapse(x_ext_main)

    # Значение цели с учётом преобразования min/max
    z = -sense_mult * tab2.v

    t1 = time.time()
    return {
        "status": "optimal",
        "x_original": x_orig,
        "objective": z,
        "iterations": it1 + it2,
        "time_sec": t1 - t0,
    }


def main():
    """Точка входа: чтение файла, запуск решения, печать результата."""
    if len(sys.argv) < 2:
        print("Использование: python main.py problem.txt")
        print("\nПример входного файла:\n")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    res = solve_from_text(text, verbose=True)
    print("\n=== РЕЗУЛЬТАТ ===")
    for k, v in res.items():
        if k == "x_original":
            print(f"{k}: {[round(x, 8) for x in v]}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()