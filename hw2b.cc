// hybrid version
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <array>
#include <algorithm>
#include <utility>

class Timer
{
#ifdef TIMER
    static const int enable = 1;
#else
    static const int enable = 0;
#endif // TIEMR
    static const int root = 0;

    // private:
public:
    double io = 0, comm = 0, cpu = 0, start, io_temp, comm_temp, cpu_temp;

public:
    void init()
    {
        if (enable)
        {
            printf("Enable timer.\n");
            start = MPI_Wtime();
        }
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

class Block
{
public:
    static const int N_BOUNDARY_INT = 4;

private:
    // int boundary[N_BOUNDARY_INT];             // store the index of the pixel block boundaries of the image
    std::array<int, N_BOUNDARY_INT> boundary; // store the index of the pixel block boundaries of the image
    int *image;                               // store #repeats

public:
    /* constructor, destructor, operator */
    static Block END_TAG() { return Block(); }
    Block() : boundary(), image(nullptr) {}
    Block(const std::array<int, N_BOUNDARY_INT> &_boundary) : boundary(_boundary)
    {
#ifdef DEBUG
        printf("const array &\n");
#endif // DEBUG
        image = (int *)malloc(size() * sizeof(int));
    }
    Block(const Block &blk) : boundary(blk.boundary)
    {
#ifdef DEBUG
        printf("const Block &\n");
#endif // DEBUG
        if (this != &blk)
        {
            free(image);
            image = (int *)malloc(blk.size() * sizeof(int));
            std::copy(blk.image, blk.image + blk.size(), image);
        }
    }
    Block(Block &&blk) : boundary(), image(nullptr)
    {
#ifdef DEBUG
        printf("Block &&, ");                                                // for debug
        printf("Before: self_image=%p, other_image=%p, ", image, blk.image); // for debug
        printf("self_boundary={");                                           // for debug
        for (size_t i = 0; i < Block::N_BOUNDARY_INT; ++i)                   // for debug
            printf("%d ", boundary[i]);                                      // for debug
        printf("}, After: ");
#endif // DEBUG

        std::swap(image, blk.image);
        boundary = std::move(blk.boundary);

#ifdef DEBUG
        printf("self_image=%p, ", image);                  // for debug
        printf("self_boundary={");                         // for debug
        for (size_t i = 0; i < Block::N_BOUNDARY_INT; ++i) // for debug
            printf("%d ", boundary[i]);                    // for debug
        printf("}\n");                                     // for debug
#endif                                                     // DEBUG
    }
    ~Block()
    {
#ifdef DEBUG
        printf("~Block free image=%p\n", image); // for debug
#endif                                           // DEBUG

        free(image);
        image = nullptr;
    }
    Block &operator=(Block &&blk)
    {
#ifdef DEBUG
        printf("= Block &&\n"); // for debug
#endif                          // DEBUG
        boundary = std::move(blk.boundary);
        std::swap(image, blk.image);
        return *this;
    }

    /* getter */
    const int *get_boundary() const { return boundary.data(); }
    int get_boundary(int idx) const { return boundary[idx]; }
    const int *get_image() const { return image; }
    int get_image(int idx) const { return image[idx]; }
    const int h0() const { return boundary[0]; }
    const int h1() const { return boundary[1]; }
    const int v0() const { return boundary[2]; }
    const int v1() const { return boundary[3]; }
    const int width() const { return h1() - h0(); }
    const int height() const { return v1() - v0(); }
    const int size() const { return (h1() - h0()) * (v1() - v0()); }
    bool is_END_TAG() const { return size() == 0; }

    /* setter */
    void set_image(int idx, int val) { image[idx] = val; }
};

class Process
{
public:
    static const int ROOT = 0;

private:
    int nproc, pid, n_cpu;
    struct Args
    {
        const char *filename;
        double left, right, lower, upper;
        int iters, width, height;

        Args(char **argv);
    } const args;

public:
    /* constructor */
    Process(char **argv);

    /* getter */
    int get_pid() const { return pid; }
    int get_nproc() const { return nproc; }
    bool is_root() const { return pid == ROOT; }

    /* args getter */
    const char *get_arg_filename() const { return args.filename; }
    double get_arg_left() const { return args.left; }
    double get_arg_right() const { return args.right; }
    double get_arg_lower() const { return args.lower; }
    double get_arg_upper() const { return args.upper; }
    int get_arg_iters() const { return args.iters; }
    int get_arg_width() const { return args.width; }
    int get_arg_height() const { return args.height; }

    /* mandelbrot set utils */
    void render(Block &blk);
};

class Scheduler
{
    static const int MASTER_NOT_COMPUTE = true;
    static const int BOUNDARY_TAG = 0;
    static const int RESULT_TAG = 1;

private:
    Process *proc = nullptr;
    int *image = nullptr;
    int res_pixels = -1;

    Block get_next_block();
    void assign_block(const Block &blk, int dest);
    Block receive_block(int src, MPI_Status *status);
    void submit_result(const Block &blk);
    Block accept_result(MPI_Status *status);

    void master_impl();

public:
    void init(Process *_proc, int *_image);
    void stitch_block(const Block &blk);
    void master();
    void servant();
    static void *pthread_master(void *scheduler);
    static void *pthread_servant(void *scheduler);
};

void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char **argv)
{
    /* MPI initialize */
    MPI_Init(&argc, &argv);
    timer.init();

    /* argument parsing */
    assert(argc == 9);
    Process proc(argv);

    /* allocate memory for image */
    int *image = nullptr;
    // pthread_t t;
    Scheduler schr;
    if (proc.is_root())
    {
        image = (int *)malloc(proc.get_arg_width() * proc.get_arg_height() * sizeof(int));
        schr.init(&proc, image);
        if (proc.get_nproc() == 1) // sequential rendering
        {
            std::array<int, Block::N_BOUNDARY_INT> boundary = {0, proc.get_arg_width(), 0, proc.get_arg_height()};
            Block blk(boundary);
            proc.render(blk);
            schr.stitch_block(blk);
        }
        else
        {
            // assert(pthread_create(&t, NULL, &Scheduler::pthread_master, &schr) == 0);
            schr.master();
        }
    }
    else
    {
        schr.init(&proc, nullptr);
        schr.servant();
    }

#ifdef TIMER
    timer.print_cpu_time();
#endif // TIMER

    /* draw and cleanup */
    MPI_Barrier(MPI_COMM_WORLD);
    if (proc.is_root())
    {
        // pthread_join(t, NULL);
        write_png(proc.get_arg_filename(), proc.get_arg_iters(), proc.get_arg_width(), proc.get_arg_height(), image);
        free(image);
    }
    MPI_Finalize();
    return 0;
}

/* Process constructor */
Process::Args::Args(char **argv) : filename(argv[1])
{
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
}
Process::Process(char **argv) : args(argv)
{
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    n_cpu = CPU_COUNT(&cpu_set);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
}

/* Process mandelbrot set utils */
void Process::render(Block &blk)
{
#ifdef DEBUG
    printf("%d rendering boundary={", pid);            // for debug
    for (size_t i = 0; i < Block::N_BOUNDARY_INT; ++i) // for debug
        printf("%d ", blk.get_boundary(i));            // for debug
    printf("}, image=%p\n", blk.get_image());          // for debug
#endif                                                 // DEBUG
    int chunk_size = (blk.size() / n_cpu) >> 1;
#pragma omp parallel for schedule(dynamic) default(shared) collapse(2)
    for (int j = blk.v0(); j < blk.v1(); j++)
    {
        for (int i = blk.h0(); i < blk.h1(); i++)
        {
            double y0 = j * ((args.upper - args.lower) / args.height) + args.lower;
            double x0 = i * ((args.right - args.left) / args.width) + args.left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < args.iters && length_squared < 4)
            {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }

            int idx = (j - blk.v0()) * blk.width() + (i - blk.h0());
            blk.set_image(idx, repeats);
        }
    }
}

/* Scheduler Private Functions */
Block Scheduler::get_next_block()
{
    static const int grand_size = proc->get_nproc() << 5;
    static const int image_width = proc->get_arg_width();
    static const int image_height = proc->get_arg_height();
    static const int blk_width = image_width;
    static const int blk_height = (image_height / grand_size + 1) * blk_width < 400 ? 400 / blk_width : (image_height / grand_size + 1);
    static const int blk_size = blk_height * blk_width;
    // static int h0 = 0;
    static int v0 = 0;
#ifdef DEBUG
    printf("\nres_pixels=%d\n", res_pixels); // for debug
#endif                                       // DEBUG
    if (res_pixels > 0)
    {
        // int h1 = h0 + blk_width < image_width ? h0 + blk_width : image_width;
        int v1 = v0 + blk_height < image_height ? v0 + blk_height : image_height;
        res_pixels -= blk_size;
        // std::array<int, Block::N_BOUNDARY_INT> boundary = {h0, h1, v0, v1};
        // h0 += blk_width;
        // if (h0 >= image_width)
        // {
        //     h0 = 0;
        //     v0 += blk_height;
        // }
        std::array<int, Block::N_BOUNDARY_INT> boundary = {0, image_width, v0, v1};
        v0 += blk_height;
#ifdef DEBUG
        for (size_t i = 0; i < Block::N_BOUNDARY_INT; ++i) // for debug
            printf("%d ", boundary[i]);                    // for debug
        printf("\n");                                      // for debug
#endif                                                     // DEBUG
        return Block(boundary);
    }
    else
        return Block::END_TAG();
}
void Scheduler::stitch_block(const Block &blk)
{
#ifdef DEBUG
    printf("stitch_block {");                          // for debug
    for (size_t i = 0; i < Block::N_BOUNDARY_INT; ++i) // for debug
        printf("%d ", blk.get_boundary(i));            // for debug
    printf("}\n");                                     // for debug
#endif                                                 // DEBUG
    int idx = 0;
    for (int j = blk.v0(); j < blk.v1(); j++)
        for (int i = blk.h0(); i < blk.h1(); i++)
            image[j * proc->get_arg_width() + i] = blk.get_image(idx++);
}
void Scheduler::assign_block(const Block &blk, int dest)
{
#ifdef DEBUG
    printf("%d assign boundary {", proc->get_pid());   // for debug
    for (size_t i = 0; i < Block::N_BOUNDARY_INT; ++i) // for debug
        printf("%d ", blk.get_boundary(i));            // for debug
    printf("} to %d\n", dest);                         // for debug
#endif                                                 // DEBUG
    // MPI_Send(blk.get_boundary(), Block::N_BOUNDARY_INT, MPI_INT, dest, BOUNDARY_TAG, MPI_COMM_WORLD);
    MPI_Request req;
    MPI_Isend(blk.get_boundary(), Block::N_BOUNDARY_INT, MPI_INT, dest, BOUNDARY_TAG, MPI_COMM_WORLD, &req);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}
Block Scheduler::receive_block(int src, MPI_Status *status)
{
    std::array<int, Block::N_BOUNDARY_INT> boundary;
    MPI_Recv(boundary.data(), Block::N_BOUNDARY_INT, MPI_INT, src, BOUNDARY_TAG, MPI_COMM_WORLD, status);
#ifdef DEBUG
    printf("%d recv boundary {", proc->get_pid());     // for debug
    for (size_t i = 0; i < Block::N_BOUNDARY_INT; ++i) // for debug
        printf("%d ", boundary[i]);                    // for debug
    printf("} from %d\n", status->MPI_SOURCE);         // for debug
#endif                                                 // DEBUG
    return Block(boundary);
}
void Scheduler::submit_result(const Block &blk)
{
#ifdef DEBUG
    printf("%d submit boundary={", proc->get_pid());                // for debug
    for (size_t i = 0; i < Block::N_BOUNDARY_INT; ++i)              // for debug
        printf("%d ", blk.get_boundary(i));                         // for debug
    for (size_t i = 0; i < blk.size(); ++i)                         // for debug
        printf("%d ", blk.get_image(i));                            // for debug
    printf("}, image=%p to %d.\n", blk.get_image(), Process::ROOT); // for debug
#endif                                                              // DEBUG
    // MPI_Send(blk.get_boundary(), Block::N_BOUNDARY_INT, MPI_INT, Process::ROOT, BOUNDARY_TAG, MPI_COMM_WORLD);
    // MPI_Send(blk.get_image(), blk.size(), MPI_INT, Process::ROOT, RESULT_TAG, MPI_COMM_WORLD);
    MPI_Request req1, req2;
    MPI_Isend(blk.get_boundary(), Block::N_BOUNDARY_INT, MPI_INT, Process::ROOT, BOUNDARY_TAG, MPI_COMM_WORLD, &req1);
    MPI_Wait(&req1, MPI_STATUS_IGNORE);
    MPI_Isend(blk.get_image(), blk.size(), MPI_INT, Process::ROOT, RESULT_TAG, MPI_COMM_WORLD, &req2);
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
}
Block Scheduler::accept_result(MPI_Status *status)
{
    Block blk = std::move(receive_block(MPI_ANY_SOURCE, status));
    MPI_Recv(const_cast<int *>(blk.get_image()), blk.size(), MPI_INT, status->MPI_SOURCE, RESULT_TAG, MPI_COMM_WORLD, status);
#ifdef DEBUG
    printf("%d accept boundary={", Process::ROOT);     // for debug
    for (size_t i = 0; i < Block::N_BOUNDARY_INT; ++i) // for debug
        printf("%d ", blk.get_boundary(i));            // for debug
    printf("}, image={");                              // for debug;
    for (size_t i = 0; i < blk.size(); ++i)            // for debug
        printf("%d ", blk.get_image(i));               // for debug
    printf("} from %d.\n", status->MPI_SOURCE);        // for debug
#endif                                                 // DEBUG
    return blk;
}

/* Scheduler Public Functions */
void Scheduler::init(Process *_proc, int *_image)
{
    proc = _proc;
    image = _image;
    res_pixels = proc->get_arg_width() * proc->get_arg_height();
}
void Scheduler::master()
{
    MPI_Status status;
    Block blk, next_blk;

    for (int pid = MASTER_NOT_COMPUTE; pid < proc->get_nproc(); pid++)
    {
        next_blk = std::move(get_next_block());
        assign_block(next_blk, pid);
    }

    for (int n_terminate = MASTER_NOT_COMPUTE; n_terminate < proc->get_nproc();)
    {
        blk = std::move(accept_result(&status));
        stitch_block(blk);
        next_blk = std::move(get_next_block());
        assign_block(next_blk, status.MPI_SOURCE);

        if (next_blk.is_END_TAG())
            ++n_terminate;
#ifdef DEBUG
        printf("N_TERMINATE=%d\n", n_terminate); // for debug
#endif                                           // DEBUG
    }
#ifdef DEBUG
    printf("%d EXIT MASTER.\n", proc->get_pid()); // for debug
#endif                                            // DEBUG
}
void Scheduler::servant()
{
    Block blk;
    MPI_Status status;

    while (true)
    {
        timer.comm_begin();
        blk = std::move(receive_block(Process::ROOT, &status));
        timer.comm_end();
#ifdef DEBUG
        printf("is_END_TAG()=%d\n", blk.is_END_TAG()); // for debug
#endif                                                 // DEBUG
        if (blk.is_END_TAG())
            break;

        timer.cpu_begin();
        proc->render(blk);
        timer.cpu_end();

        timer.comm_begin();
        submit_result(blk);
        timer.comm_end();
    }
#ifdef DEBUG
    printf("%d EXIT SERVANT.\n", proc->get_pid()); // for debug
#endif                                             // DEBUG
}
void *Scheduler::pthread_master(void *scheduler)
{
    static_cast<Scheduler *>(scheduler)->master();
    return NULL;
}
void *Scheduler::pthread_servant(void *scheduler)
{
    static_cast<Scheduler *>(scheduler)->servant();
    return NULL;
}

/* Timer Functions */
void Timer::print_comm_time()
{
    if (!enable)
        return;

    int pid = -1, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&comm, &comm_temp, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if (pid == root)
        printf("Comm time:\t%f\n", comm_temp / nproc);
}
void Timer::print_io_time()
{
    if (!enable)
        return;

    int pid = -1, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&io, &io_temp, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if (pid == root)
        printf("IO time:\t%f\n", io_temp / nproc);
}
void Timer::print_cpu_time()
{
    if (!enable)
        return;

    /* int pid = -1, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&cpu, &cpu_temp, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if (pid == root)
        printf("CPU time:\t%f\n", cpu_temp / nproc); */
    printf("CPU time:\t%f\n", cpu); // self cpu time
}
