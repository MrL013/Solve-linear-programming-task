import numpy as np
import re

class SimplexSolver:
    def __init__(self):
        self.A = None
        self.b = None
        self.c = None
        self.basis = None
        self.artificial_vars = []
        self.problem_type = None
        self.num_vars = 0
        self.num_constraints = 0

    def parse_problem(self, text):
        """Парсинг текстового описания задачи"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Определение типа задачи
        if 'Максимизировать' in lines[0] or 'максимизировать' in lines[0]:
            self.problem_type = 'max'
        else:
            self.problem_type = 'min'

        # Определяем количество переменных
        self.num_vars = 4  # x1, x2, x3, x4

        # Извлечение целевой функции
        objective_match = re.search(r'[Zz]\s*=\s*([\d\.\-\+\*x\s_]+)', lines[0])
        if objective_match:
            objective_str = objective_match.group(1)
            self.c_original = self.parse_expression(objective_str, self.num_vars)

        # Парсинг ограничений
        constraints = []
        constraint_types = []

        for line in lines[1:]:
            if '≤' in line or '<=' in line:
                constraint_types.append('<=')
                coeffs, rhs = self.parse_constraint(line, self.num_vars)
                constraints.append((coeffs, rhs))
            elif '≥' in line or '>=' in line:
                constraint_types.append('>=')
                coeffs, rhs = self.parse_constraint(line, self.num_vars)
                constraints.append((coeffs, rhs))
            elif '=' in line:
                constraint_types.append('=')
                coeffs, rhs = self.parse_constraint(line, self.num_vars)
                constraints.append((coeffs, rhs))

        self.num_constraints = len(constraints)
        return constraints, constraint_types

    def parse_expression(self, expr, num_vars):
        """Парсинг математического выражения"""
        coeffs = [0.0] * num_vars

        # Ищем все слагаемые с переменными
        pattern = r'([+-]?\s*\d*\.?\d*)\s*\*?\s*x\s*_?\s*(\d+)'
        matches = re.findall(pattern, expr)

        for coeff_str, var_idx in matches:
            idx = int(var_idx) - 1
            if idx >= num_vars:
                continue

            coeff_str = coeff_str.replace(' ', '')

            if coeff_str == '' or coeff_str == '+':
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)

            coeffs[idx] = coeff

        return coeffs

    def parse_constraint(self, line, num_vars):
        """Парсинг ограничения"""
        # Определяем тип ограничения
        if '≤' in line:
            parts = line.split('≤')
        elif '>=' in line:
            parts = line.split('>=')
        elif '≥' in line:
            parts = line.split('≥')
        elif '=' in line:
            parts = line.split('=')
        else:
            return None, None

        left_side = parts[0].strip()
        right_side = float(parts[1].strip())

        coeffs = self.parse_expression(left_side, num_vars)
        return coeffs, right_side

    def to_canonical_form(self, constraints, constraint_types):
        """Приведение к канонической форме"""
        print("\n=== ПРИВЕДЕНИЕ К КАНОНИЧЕСКОЙ ФОРМЕ ===")

        # Подсчитываем количество дополнительных переменных
        slack_count = 0
        artificial_count = 0

        for constr_type in constraint_types:
            if constr_type == '<=':
                slack_count += 1
            elif constr_type == '>=':
                slack_count += 1  # surplus переменная
                artificial_count += 1
            elif constr_type == '=':
                artificial_count += 1

        self.total_slack = slack_count
        self.total_artificial = artificial_count
        total_additional = slack_count + artificial_count
        self.total_vars = self.num_vars + total_additional

        print(f"Исходных переменных: {self.num_vars}")
        print(f"Slack/surplus переменных: {slack_count}")
        print(f"Искусственных переменных: {artificial_count}")
        print(f"Всего переменных: {self.total_vars}")

        # Создаем расширенную матрицу A
        A_extended = []
        b_list = []

        slack_idx = 0
        artificial_idx = 0
        self.artificial_vars = []

        print("\nОграничения в канонической форме:")
        for i, ((coeffs, rhs), constr_type) in enumerate(zip(constraints, constraint_types)):
            row = coeffs.copy()
            constraint_str = f"Ограничение {i + 1}: "

            # Добавляем slack/surplus/artificial переменные
            additional_vars = [0.0] * total_additional

            if constr_type == '<=':
                # Slack переменная
                additional_vars[slack_idx] = 1.0
                constraint_str += " + ".join([f"{coeffs[j]}x_{j + 1}" for j in range(len(coeffs)) if coeffs[j] != 0])
                constraint_str += f" + s_{slack_idx + 1} = {rhs}"
                slack_idx += 1
            elif constr_type == '>=':
                # Surplus и artificial переменные
                additional_vars[slack_idx] = -1.0  # surplus
                artificial_var_idx = slack_count + artificial_idx
                additional_vars[artificial_var_idx] = 1.0  # artificial
                self.artificial_vars.append(self.num_vars + artificial_var_idx)
                constraint_str += " + ".join([f"{coeffs[j]}x_{j + 1}" for j in range(len(coeffs)) if coeffs[j] != 0])
                constraint_str += f" - s_{slack_idx + 1} + a_{artificial_idx + 1} = {rhs}"
                slack_idx += 1
                artificial_idx += 1
            else:  # '='
                # Artificial переменная
                artificial_var_idx = slack_count + artificial_idx
                additional_vars[artificial_var_idx] = 1.0
                self.artificial_vars.append(self.num_vars + artificial_var_idx)
                constraint_str += " + ".join([f"{coeffs[j]}x_{j + 1}" for j in range(len(coeffs)) if coeffs[j] != 0])
                constraint_str += f" + a_{artificial_idx + 1} = {rhs}"
                artificial_idx += 1

            full_row = row + additional_vars
            A_extended.append(full_row)
            b_list.append(rhs)
            print(constraint_str)

        self.A = np.array(A_extended, dtype=float)
        self.b = np.array(b_list, dtype=float)

        # Целевая функция для канонической формы (всегда минимизация)
        if self.problem_type == 'max':
            c_extended = [-x for x in self.c_original] + [0.0] * total_additional
        else:
            c_extended = self.c_original + [0.0] * total_additional

        self.c = np.array(c_extended, dtype=float)

        print(f"\nЦелевая функция в канонической форме: min {self.c}")
        print("Матрица A:")
        print(self.A)
        print("Вектор b:", self.b)
        print("Искусственные переменные:", [f"x{i + 1}" for i in self.artificial_vars])

    def find_initial_basis_phase1(self):
        """Нахождение начального базиса для фазы 1"""
        basis = []

        # Сначала добавляем искусственные переменные
        for art_var in self.artificial_vars:
            basis.append(art_var)

        # Если нужно больше базисных переменных, добавляем slack переменные
        for j in range(self.num_vars, self.num_vars + self.total_slack):
            if j not in basis and len(basis) < self.num_constraints:
                # Проверяем, можно ли добавить эту переменную в базис
                col = self.A[:, j]
                # Проверяем, что столбец линейно независим от уже выбранных
                if len(basis) > 0:
                    current_basis_matrix = self.A[:, basis]
                    try:
                        # Пробуем решить систему чтобы проверить линейную независимость
                        extended_matrix = np.column_stack([current_basis_matrix, col])
                        if np.linalg.matrix_rank(extended_matrix) > np.linalg.matrix_rank(current_basis_matrix):
                            basis.append(j)
                    except:
                        basis.append(j)
                else:
                    basis.append(j)

        print(f"Начальный базис фазы 1: {[f'x{i + 1}' for i in basis]}")
        return basis

    def phase1_simplex(self):
        """Фаза 1: минимизация суммы искусственных переменных"""
        print("\n=== ФАЗА 1 ===")

        # Создаем целевую функцию для фазы 1
        c_phase1 = np.zeros(self.total_vars)
        for art_var in self.artificial_vars:
            c_phase1[art_var] = 1.0

        print(f"Целевая функция фазы 1: {c_phase1}")

        # Начальный базис
        basis = self.find_initial_basis_phase1()

        # Решаем симплекс-методом для фазы 1
        solution, final_basis, msg = self.simplex_core(c_phase1, basis, "Фаза 1")

        if solution is None:
            return None, None, msg

        # Проверяем, что все искусственные переменные равны 0
        artificial_sum = sum(solution[i] for i in self.artificial_vars)
        print(f"Сумма искусственных переменных: {artificial_sum}")

        if artificial_sum > 1e-6:
            return None, None, "Область допустимых решений пуста"

        # Убираем искусственные переменные из базиса если возможно
        clean_basis = []
        for var in final_basis:
            if var not in self.artificial_vars:
                clean_basis.append(var)

        # Если нужно, добавляем другие переменные чтобы получить полный базис
        while len(clean_basis) < self.num_constraints:
            for j in range(self.total_vars):
                if j not in clean_basis and j not in self.artificial_vars:
                    clean_basis.append(j)
                    break

        return solution, clean_basis, "Фаза 1 завершена успешно"

    def phase2_simplex(self, initial_basis):
        """Фаза 2: решение исходной задачи"""
        print("\n=== ФАЗА 2 ===")
        print(f"Начальный базис: {[f'x{i + 1}' for i in initial_basis]}")

        solution, final_basis, msg = self.simplex_core(self.c, initial_basis, "Фаза 2")

        if solution is None:
            return None, None, msg

        return solution, final_basis, "Фаза 2 завершена успешно"

    def simplex_core(self, c, initial_basis, phase_name):
        """Ядро симплекс-метода"""
        basis = initial_basis.copy()
        max_iterations = 50

        for iteration in range(max_iterations):
            print(f"\n{phase_name} - Итерация {iteration + 1}")
            print(f"Базис: {[f'x{i + 1}' for i in basis]}")

            # 1. Вычисляем базисное решение
            A_b = self.A[:, basis]

            # Проверяем, что матрица квадратная и невырожденная
            if A_b.shape[0] != A_b.shape[1]:
                print(f"Ошибка: матрица A_b не квадратная {A_b.shape}")
                return None, None, "Матрица не квадратная"

            try:
                # Проверяем определитель
                det = np.linalg.det(A_b)
                if abs(det) < 1e-10:
                    print("Вырожденная матрица")
                    # Пробуем найти другой базис
                    for j in range(self.total_vars):
                        if j not in basis:
                            # Пробуем заменить одну переменную в базисе
                            for i in range(len(basis)):
                                test_basis = basis.copy()
                                test_basis[i] = j
                                test_A_b = self.A[:, test_basis]
                                if abs(np.linalg.det(test_A_b)) > 1e-10:
                                    basis = test_basis
                                    A_b = test_A_b
                                    print(f"Исправлен базис: {[f'x{i + 1}' for i in basis]}")
                                    break
                            break

                x_b = np.linalg.solve(A_b, self.b)
            except np.linalg.LinAlgError:
                print("Матрица вырожденная, не удалось решить систему")
                return None, None, "Вырожденная матрица"

            print(f"Базисное решение: {x_b}")

            # 2. Вычисляем оценки для небазисных переменных
            non_basis = [j for j in range(self.total_vars) if j not in basis]
            reduced_costs = []

            c_b = c[basis]

            for j in non_basis:
                # Вычисляем z_j = c_B * B^{-1} * A_j
                A_j = self.A[:, j]
                # Решаем систему B * y = A_j
                try:
                    y = np.linalg.solve(A_b, A_j)
                    z_j = np.dot(c_b, y)
                    reduced_cost = c[j] - z_j
                    reduced_costs.append((j, reduced_cost))
                except:
                    reduced_costs.append((j, float('inf')))

            # 3. Проверка оптимальности
            if all(rc[1] >= -1e-6 for rc in reduced_costs):
                print("Достигнуто оптимальное решение!")
                # Формируем полное решение
                x_full = np.zeros(self.total_vars)
                for i, var_idx in enumerate(basis):
                    x_full[var_idx] = x_b[i]
                return x_full, basis, "Оптимально"

            # 4. Выбор вводимой переменной (наименьшая оценка)
            valid_reduced_costs = [rc for rc in reduced_costs if rc[1] < -1e-6 and not np.isinf(rc[1])]
            if not valid_reduced_costs:
                print("Нет переменных с отрицательной оценкой")
                x_full = np.zeros(self.total_vars)
                for i, var_idx in enumerate(basis):
                    x_full[var_idx] = x_b[i]
                return x_full, basis, "Оптимально"

            entering_var, min_rc = min(valid_reduced_costs, key=lambda x: x[1])
            print(f"Вводимая переменная: x{entering_var + 1}, оценка: {min_rc:.6f}")

            # 5. Вычисление направления
            try:
                d = np.linalg.solve(A_b, self.A[:, entering_var])
            except:
                print("Ошибка при вычислении направления")
                return None, None, "Ошибка вычисления"

            # 6. Проверка на неограниченность
            if all(d_i <= 1e-10 for d_i in d):
                print("Задача неограниченна!")
                return None, None, "Неограниченна"

            # 7. Выбор исключаемой переменной
            ratios = []
            for i in range(len(basis)):
                if d[i] > 1e-10:
                    ratio = x_b[i] / d[i]
                    ratios.append((i, ratio, basis[i]))
                else:
                    ratios.append((i, float('inf'), basis[i]))

            # Убираем бесконечные отношения
            valid_ratios = [r for r in ratios if r[1] != float('inf')]
            if not valid_ratios:
                print("Нет допустимого решения!")
                return None, None, "Нет допустимого решения"

            leaving_idx, min_ratio, leaving_var = min(valid_ratios, key=lambda x: x[1])
            print(f"Исключаемая переменная: x{leaving_var + 1}, отношение: {min_ratio:.6f}")

            # 8. Обновление базиса
            basis[leaving_idx] = entering_var

        print("Превышено максимальное число итераций!")
        return None, None, "Превышено число итераций"

    def solve(self, text):
        """Основной метод решения"""
        try:
            # Парсинг задачи
            constraints, constraint_types = self.parse_problem(text)

            print("Исходная задача:")
            print(f"Тип: {self.problem_type}")
            print(f"Целевая функция: Z = {self.c_original}")
            print("Ограничения:")
            for i, ((coeffs, rhs), constr_type) in enumerate(zip(constraints, constraint_types)):
                constraint_str = " + ".join([f"{coeffs[j]}x_{j + 1}" for j in range(len(coeffs)) if coeffs[j] != 0])
                print(f"  {constraint_str} {constr_type} {rhs}")

            # Приведение к канонической форме
            self.to_canonical_form(constraints, constraint_types)

            # Двухфазный симплекс-метод
            if self.artificial_vars:
                # Фаза 1
                phase1_solution, phase1_basis, phase1_msg = self.phase1_simplex()
                if phase1_solution is None:
                    return {
                        'status': 'error',
                        'message': phase1_msg,
                        'optimal_point': None,
                        'optimal_value': None
                    }

                print(f"Фаза 1 успешно завершена. Базис для фазы 2: {[f'x{i + 1}' for i in phase1_basis]}")

                # Фаза 2
                solution, basis, msg = self.phase2_simplex(phase1_basis)
            else:
                # Если нет искусственных переменных, сразу фаза 2
                initial_basis = self.find_initial_basis_phase1()
                solution, basis, msg = self.phase2_simplex(initial_basis)

            if solution is None:
                return {
                    'status': 'error',
                    'message': msg,
                    'optimal_point': None,
                    'optimal_value': None
                }

            # Извлекаем только исходные переменные
            original_solution = solution[:self.num_vars]

            # Вычисляем значение целевой функции
            if self.problem_type == 'max':
                optimal_value = np.dot(self.c_original, original_solution)
            else:
                optimal_value = np.dot(self.c_original, original_solution)

            return {
                'status': 'success',
                'message': msg,
                'optimal_point': original_solution,
                'optimal_value': optimal_value,
                'basis': basis
            }

        except Exception as e:
            import traceback
            return {
                'status': 'error',
                'message': f'Ошибка: {str(e)}\n{traceback.format_exc()}',
                'optimal_point': None,
                'optimal_value': None
            }


def create_problem_file():
    """Создание файла с задачей"""
    problem_text = """Максимизировать Z = 4x_1 + x_2 + 2x_3 + 3x_4

x_1 + x_2 + x_3 <= 9
x_2 + 2x_3 + x_4 = 7
x_1 + x_4 >= 3"""

    with open('problem.txt', 'w', encoding='utf-8') as f:
        f.write(problem_text)

    return problem_text


def main():
    """Основная функция"""
    # Создаем файл с задачей
    problem_text = create_problem_file()

    # Создаем решатель
    solver = SimplexSolver()

    # Решаем задачу
    print("=" * 60)
    print("РЕШЕНИЕ ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")
    print("=" * 60)

    result = solver.solve(problem_text)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)

    if result['status'] == 'success':
        print("✓ Решение найдено!")
        print(f"Оптимальное значение: Z = {result['optimal_value']:.6f}")
        print("Оптимальная точка:")
        for i, x_val in enumerate(result['optimal_point']):
            print(f"  x_{i + 1} = {x_val:.6f}")

        print(f"Базисные переменные: {[f'x{i + 1}' for i in result['basis']]}")

        # Проверка ограничений
        print("\nПроверка ограничений:")
        x = result['optimal_point']
        constraint1 = x[0] + x[1] + x[2]
        constraint2 = x[1] + 2 * x[2] + x[3]
        constraint3 = x[0] + x[3]

        print(f"x₁ + x₂ + x₃ = {constraint1:.6f} ≤ 9 ({constraint1 <= 9 + 1e-6})")
        print(f"x₂ + 2x₃ + x₄ = {constraint2:.6f} = 7 ({abs(constraint2 - 7) < 1e-6})")
        print(f"x₁ + x₄ = {constraint3:.6f} ≥ 3 ({constraint3 >= 3 - 1e-6})")

    else:
        print("✗ Решение не найдено")
        print(f"Причина: {result['message']}")


if __name__ == "__main__":
    main()