#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <mpi.h>

// Вычисление q_{b+1}(x) = a_{br+1} + a_{br+2}*x + ... + a_{br+r}*x^{r-1}
double compute_block(const std::vector<double>& a, int b, int r, double x) {
    double result = 0.0;
    double x_pow = 1.0;

    // Индекс a_{br+1}
    int base = b * r + 1;
    
    for (int j = 0; j < r; ++j) {
        result += a[base + j] * x_pow;
        x_pow *= x;
    
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Параметры задачи
    int r = 6;            // Размер блока (число коэффициентов в блоке)
    double x = 1.2;       // Точка, в которой вычисляется многочлен
    if (argc > 1) r = std::atoi(argv[1]); // mpirun -n 4 ./bin/var_1.cpp r 1.1 
    if (argc > 2) x = std::atof(argv[2]); // mpirun -n 4 ./bin/var_1 6 x
    int s = 1 << r;       // Число блоков, s = 2^r — степень двойки
    int n = s * r + 1;    // Число коэффициентов: a_0 ... a_{sr}

    // Коэффициенты
    std::vector<double> a(n);
    for (int i = 0; i < n; ++i) {
        a[i] = 1.0 + 0.01 * i;
    }

    // Последовательный эталон
    double q_ref = 0.0;
    if (rank == 0) {
        q_ref = a[0];
        double xr = 1.0;
        for (int j = 0; j < r; ++j) xr *= x;
        double factor = 1.0;
        for (int i = 1; i <= s; ++i) {
            double qi = 0.0;
            double x_pow = 1.0;
            int base = (i - 1) * r + 1;
            for (int j = 0; j < r; ++j) {
                qi += a[base + j] * x_pow;
                x_pow *= x;
            }
            q_ref += factor * qi;
            factor *= xr;
        }
    }

    // Параллельные вычисления
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Вычисление x^r по одному разу на каждом процессе
    double xr = 1.0;
    for (int j = 0; j < r; ++j) xr *= x;

    // Блочное распределение блоков b = 0...s-1 между процессами
    int base_blocks = s / size;
    int rem_blocks  = s % size;
    int start_b, end_b;
    if (rank < rem_blocks) {
        // Первым rem_blocks процессам достаётся на один блок больше
        start_b = rank * (base_blocks + 1);
        end_b   = start_b + base_blocks;
    } else {
        start_b = rem_blocks * (base_blocks + 1) + (rank - rem_blocks) * base_blocks;
        end_b   = start_b + base_blocks - 1;
    }

    // Подготовка множителя x^{start_b*r} (последовательное вычисление)
    double factor = 1.0;
    for (int k = 0; k < start_b; ++k) factor *= xr;

    // Вычисление блоков и их вкладов
    double local_sum = 0.0;
    for (int b = start_b; b <= end_b; ++b) {
        double qi = compute_block(a, b, r, x);  // q_{b+1}(x)
        local_sum += factor * qi;
        factor *= xr;
    }

    // Суммирование вкладов всех процессов
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    if (rank == 0) {
        double q_par = a[0] + global_sum;
        std::cout << "=== Вычисление значения многочлена q(x) ===\n";
        std::cout << "r = " << r << ", s = 2^r = " << s
                  << ", n = s*r+1 = " << n << "\n";
        std::cout << "x = " << x << ", процессов = " << size << "\n";
        std::cout << "Эталон (последовательный): " << q_ref << "\n";
        std::cout << "Параллельный результат   : " << q_par << "\n";
        std::cout << "Абсолютная разница       : "
                  << std::fabs(q_ref - q_par) << "\n";
        std::cout << "Относительная разница    : "
                  << std::fabs(q_ref - q_par) / std::fabs(q_ref) << "\n";
        std::cout << "Время: " << (t_end - t_start) * 1e6 << " мкс\n";
    }

    MPI_Finalize();
    return 0;
}
