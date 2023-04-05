#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <mpi.h>
#include <algorithm>
using std::copy;
using std::stable_sort;
using std::swap;

#define PAUSE                                   \
    printf("Press Enter key to continue...\n"); \
    fgetc(stdin);

int ndata = 0;
const int comm_cost = 10000;

class Timer
{
    static const int enable = 0;
    static const int root = 0;

// private:
public:
    double io = 0, comm = 0, cpu = 0, start, io_temp, comm_temp, cpu_temp;

public:
    void init()
    {
        if (enable)
            start = MPI_Wtime();
    }
    void io_begin()
    {
        if (enable)
            io_temp = MPI_Wtime();
    }
    void io_end()
    {
        if (enable)
            io += MPI_Wtime() - io_temp;
    }
    void comm_begin()
    {
        if (enable)
            comm_temp = MPI_Wtime();
    }
    void comm_end()
    {
        if (enable)
            comm += MPI_Wtime() - comm_temp;
    }
    void cpu_begin()
    {
        if (enable)
            cpu_temp = MPI_Wtime();
    }
    void cpu_end()
    {
        if (enable)
            cpu += MPI_Wtime() - cpu_temp;
    }
    void print_comm_time();
    void print_io_time();
    void print_cpu_time();
} timer;

class Proc
{
private:
    const int tag = 0;

public:
    int nproc = 1;
    float *data, *mailbox;
    int pid, size;
    MPI_Request req;

    Proc() = default;
    ~Proc()
    {
        delete[] data;
        delete[] mailbox;
    }
    int is_first_proc() const { return pid == 0; }
    int is_last_proc() const { return pid == nproc - 1; }
    float &minimum() const { return data[0]; }        // data must be sorted
    float &maximum() const { return data[size - 1]; } // data must be sorted
    float &operator[](int i) { return data[i]; }
    void malloc(int _size);
    void print_data();  // for debug
    void display_all(); // for debug
    void send(int dest, float *arr);
    int recv(int src);
    void merge(float *merge_temp);
    void merge(float *merge_temp, int &i, int &j);
} proc;

void printarr(const char *title, float *arr, int size); // for debug
inline int worth_parallel();
inline int calc_buffer_size();
void mpi_init(int argc, char **argv);
void mpi_read(const char *input_filename);
void mpi_write(const char *output_filename);
void odd_even_parallel();

int main(int argc, char **argv)
{
    mpi_init(argc, argv);
    timer.init();

    ndata = atoi(argv[1]);
    proc.malloc(calc_buffer_size());

    // printf("ndata=%d, pid=%d, nproc=%d, count=%d\n", ndata, proc.pid, proc.nproc, proc.size); // for debug

    mpi_read(argv[2]);

    if (proc.size > 0)
    {
        timer.cpu_begin();
        stable_sort(proc.data, proc.data + proc.size);
        timer.cpu_end();
    }

    // printarr("data", proc.data, proc.size);       // for debug
    // printarr("mailbox", proc.mailbox, proc.size); // for debug

    if (worth_parallel())
        odd_even_parallel();

    mpi_write(argv[3]);

    timer.print_cpu_time();
    timer.print_comm_time();
    timer.print_io_time();

    MPI_Finalize();
    return 0;
}

// =========================== FUNCTION ===========================
void printarr(const char *title, float *arr, int size) // for debug
{
    printf("RANK=%d, %s\n", proc.pid, title);
    for (size_t i = 0; i < size; ++i)
        printf("%f ", arr[i]);
    printf("\n\n");
    PAUSE
}

inline int worth_parallel()
{
    return proc.nproc * comm_cost < ndata && proc.nproc > 1;
}

inline int calc_buffer_size()
{
    int ndata_div_nproc = ndata / proc.nproc;
    int ndata_mod_nproc = ndata % proc.nproc;

    if (worth_parallel())
    {
        if (ndata_mod_nproc)
            return ndata_div_nproc + 1;
        else
            return ndata_div_nproc;
    }
    else
    {
        if (proc.is_first_proc())
            return ndata;
        else
            return 0;
    }
}

void mpi_init(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc.pid);
    MPI_Comm_size(MPI_COMM_WORLD, &proc.nproc);
}

void mpi_read(const char *input_filename)
{
    timer.io_begin();
    const int ndata_mod_nproc = ndata % proc.nproc;
    const int offset = sizeof(float) * proc.pid * proc.size;
    const int padded_chunk_real_size = proc.size - proc.nproc + ndata_mod_nproc;
    MPI_File input_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if (ndata_mod_nproc && proc.is_last_proc() && worth_parallel())
    {
        MPI_File_read_at(input_file, offset, proc.data, padded_chunk_real_size, MPI_FLOAT, MPI_STATUS_IGNORE);
        for (size_t i = padded_chunk_real_size; i < proc.size; ++i) // padding
            proc[i] = FLT_MAX;
    }
    else
        MPI_File_read_at(input_file, offset, proc.data, proc.size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    timer.io_end();
}

void mpi_write(const char *output_filename)
{
    timer.io_begin();
    const int ndata_mod_nproc = ndata % proc.nproc;
    const int offset = sizeof(float) * proc.pid * proc.size;
    const int padded_chunk_real_size = proc.size - proc.nproc + ndata_mod_nproc;
    MPI_File output_file;
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if (ndata_mod_nproc && proc.is_last_proc() && worth_parallel())
        MPI_File_write_at(output_file, offset, proc.data, padded_chunk_real_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    else
        MPI_File_write_at(output_file, offset, proc.data, proc.size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    timer.io_end();
}

void odd_even_parallel()
{
    int pass = 0, sorted;
    // float *temp = new float[proc.size * 2]; // Version 1
    float *upper = new float[proc.size]; // Version 2
    float *lower = new float[proc.size]; // Version 2
    float *min = new float[proc.nproc]();
    float *max = new float[proc.nproc]();

    // 1. odd send, even recv
    // 2. even merge and send back
    // 3. odd recv and update data
    // 4. do another phase
    // 5. check if sorting is done
    do
    {
        for (int phase = 1; phase >= 0; phase--)
        {
            if ((proc.pid & 0x01) == phase)
            {
                proc.send(proc.pid - 1, proc.data);
                if (proc.recv(proc.pid - 1))
                    swap(proc.data, proc.mailbox);
            }
            else
            {
                if (proc.recv(proc.pid + 1))
                {
                    // Version 1
                    // proc.merge(temp);
                    // proc.send(proc.pid + 1, temp + proc.size); // send back the larger half
                    // copy(temp, temp + proc.size, proc.data);   // keep the smaller half
                
                    // Version 2
                    int i = proc.size - 1, j = proc.size - 1;
                    proc.merge(upper, i, j);
                    proc.send(proc.pid + 1, upper); // send back the larger half
                    proc.merge(lower, i, j);
                    swap(lower, proc.data); // keep the smaller half
                }
            }
        }

        timer.comm_begin();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(&proc.minimum(), 1, MPI_FLOAT, min, 1, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(&proc.maximum(), 1, MPI_FLOAT, max, 1, MPI_FLOAT, MPI_COMM_WORLD);
        timer.comm_end();

        // printarr("===MAX===", max, proc.nproc); // for debug
        // printarr("===MIN===", min, proc.nproc); // for debug

        sorted = 1;
        for (size_t i = 1; i < proc.nproc; ++i)
            sorted &= min[i] >= max[i - 1];

        // if (proc.is_first_proc())        // for debug
        //     printf("PASS %d\n", pass++); // for debug
    } while (!sorted);

    // delete[] temp; // Version 1
    delete[] upper; // Version 2
    delete[] lower; // Version 2
    delete[] min;
    delete[] max;
}

// =========================== TIMER ===========================
void Timer::print_comm_time()
{
    if (!enable)
        return;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&comm, &comm_temp, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if (proc.pid == root)
        printf("Comm time:\t%f\n", comm_temp / proc.nproc);
}
void  Timer::print_io_time()
{
    if (!enable)
        return;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&io, &io_temp, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if (proc.pid == root)
        printf("IO time:\t%f\n", io_temp / proc.nproc);
}
void Timer::print_cpu_time()
{
    if (!enable)
        return;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&cpu, &cpu_temp, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if (proc.pid == root)
        printf("CPU time:\t%f\n", cpu_temp / proc.nproc);
}

// =========================== PROC ===========================
void Proc::malloc(int _size)
{
    data = new float[_size]();
    mailbox = new float[_size]();
    size = _size;
}
void Proc::print_data() // for debug
{
    printf("pid=%d\n", pid);
    for (size_t i = 0; i < size; ++i)
        if (data[i] != FLT_MAX)
            printf("%e ", data[i]);
    printf("\n\n");
}
void Proc::display_all() // for debug
{
    MPI_Request req;
    if (pid == 0)
    {
        int *buf_size = new int[nproc];
        buf_size[0] = size;
        for (size_t i = 1; i < nproc; ++i)
            MPI_Recv(&buf_size[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        float **buf = new float *[nproc];
        buf[0] = data;
        for (size_t i = 1; i < nproc; ++i)
        {
            buf[i] = new float[buf_size[i]];
            MPI_Recv(buf[i], buf_size[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (size_t i = 0; i < nproc; ++i)
        {
            printf("====================\npid = %d\n", i);
            for (int j = 0; j < buf_size[i]; ++j)
                printf("%3d %e\n", j, buf[i][j]);
            printf("====================\n");
        }

        delete[] buf_size;
        for (size_t i = 1; i < nproc; ++i)
            delete[] buf[i];
        delete[] buf;
    }
    else
    {
        MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(data, size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
}
void Proc::send(int dest, float *arr)
{
    timer.comm_begin();
    if (dest >= 0 && dest < nproc && dest != pid)
    {
        // printf("%d send to %d\n", pid, dest); // for debug
        MPI_Isend(arr, size, MPI_FLOAT, dest, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        // MPI_Send(arr, size, MPI_FLOAT, dest, tag, MPI_COMM_WORLD); // slower
    }
    timer.comm_end();
}
int Proc::recv(int src)
{
    timer.comm_begin();
    if (src >= 0 && src < nproc && src != pid)
    {
        // printf("%d recv from %d\n", pid, src); // for debug
        MPI_Irecv(mailbox, size, MPI_FLOAT, src, tag, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        // MPI_Recv(mailbox, size, MPI_FLOAT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // slower
        timer.comm_end();
        return 1;
    }
    timer.comm_end();
    return 0;
}
void Proc::merge(float *merge_temp)
{
    timer.cpu_begin();
    int i, j, k;
    float *a = data, *b = mailbox, *t = merge_temp;
    for (i = j = size - 1, k = 2 * size - 1; k >= 0; --k)
    {
        if (i < 0)
            t[k] = b[j--];
        else if (j < 0)
            t[k] = a[i--];
        else
            t[k] = a[i] > b[j] ? a[i--] : b[j--];
    }
    // printarr("===DATA===", a, size);      // for debug
    // printarr("===MAIL===", b, size);      // for debug
    // printarr("===MERGE===", t, 2 * size); // for debug
    timer.cpu_end();
}
void Proc::merge(float *merge_temp, int &i, int &j)
{
    timer.cpu_begin();
    float *a = data, *b = mailbox, *t = merge_temp;
    for (int k = size - 1; k >= 0; --k)
    {
        if (i < 0)
            t[k] = b[j--];
        else if (j < 0)
            t[k] = a[i--];
        else
            t[k] = a[i] > b[j] ? a[i--] : b[j--];
    }
    timer.cpu_end();
}
